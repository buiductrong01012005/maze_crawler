"""Training loop for DRL agent using custom PPO."""

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig, RewardConfig
from .env_wrapper import CrawlGymnasiumEnv
from .model import CrawlActorCritic
from .obs_utils import extract_local_view, extract_scalars
from .action_utils import compute_action_mask, ACTION_ID_TO_STRING
from .reward import RewardShaper
from .evaluate import Evaluator
from .self_play import SelfPlayLeague

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Plotting is optional; CSV and TensorBoard still work.
    plt = None


class TrainingLogger:
    """Writes structured training metrics to CSV and optional PNG charts."""

    CSV_FIELDS = [
        "step",
        "updates",
        "elapsed_sec",
        "fps",
        "rollout_shaped_reward_mean",
        "rollout_raw_reward_mean",
        "episode_return_mean_100",
        "episode_length_mean_100",
        "win_rate_100",
        "loss_rate_100",
        "draw_rate_100",
        "avg_units_per_step",
        "avg_valid_actions",
        "move_north_rate",
        "build_rate",
        "mine_transform_rate",
        "transfer_rate",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "total_loss",
        "learning_rate",
        "eval_win_rate",
        "eval_avg_reward",
        "eval_avg_steps",
    ]

    def __init__(self, csv_path: str, plot_dir: str):
        self.csv_path = Path(csv_path)
        self.plot_dir = Path(plot_dir)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def log_row(self, row: Dict[str, float]):
        clean_row = {field: row.get(field, "") for field in self.CSV_FIELDS}
        self.history.append(clean_row)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(clean_row)

    def plot(self):
        if plt is None or not self.history:
            return

        steps = np.array([float(row["step"]) for row in self.history], dtype=float)

        def values(key: str):
            out = []
            for row in self.history:
                value = row.get(key, "")
                out.append(np.nan if value == "" else float(value))
            return np.array(out, dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Maze Crawler DRL Training Overview")

        axes[0, 0].plot(steps, values("episode_return_mean_100"), label="episode return")
        axes[0, 0].plot(steps, values("rollout_shaped_reward_mean"), label="rollout shaped")
        axes[0, 0].set_title("Reward")
        axes[0, 0].set_xlabel("env steps")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, values("win_rate_100"), label="train win rate")
        axes[0, 1].plot(steps, values("loss_rate_100"), label="train loss rate")
        axes[0, 1].plot(steps, values("eval_win_rate"), label="eval win rate")
        axes[0, 1].set_title("Outcomes")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_xlabel("env steps")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(steps, values("policy_loss"), label="policy")
        axes[1, 0].plot(steps, values("value_loss"), label="value")
        axes[1, 0].plot(steps, values("entropy_loss"), label="entropy")
        axes[1, 0].set_title("Losses")
        axes[1, 0].set_xlabel("env steps")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, values("fps"), label="FPS")
        axes[1, 1].plot(steps, values("avg_units_per_step"), label="avg units")
        axes[1, 1].plot(steps, values("avg_valid_actions"), label="avg valid actions")
        axes[1, 1].set_title("Runtime / Complexity")
        axes[1, 1].set_xlabel("env steps")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.plot_dir / "training_overview.png", dpi=150)
        plt.close(fig)

    def plot_action_distribution(self, action_counts: Dict[int, int], step: int):
        if plt is None or not action_counts:
            return

        action_ids = sorted(ACTION_ID_TO_STRING.keys())
        labels = [ACTION_ID_TO_STRING[action_id] for action_id in action_ids]
        counts = [action_counts.get(action_id, 0) for action_id in action_ids]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(labels, counts)
        ax.set_title(f"Action Distribution at Step {step:,}")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "action_distribution.png", dpi=150)
        plt.close(fig)


class RolloutBuffer:
    """Experience replay buffer for PPO."""

    def __init__(
        self, num_steps: int, batch_size: int, device: str = "cpu"
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device

        self.reset()

    def reset(self):
        self.local_maps = []
        self.scalars = []
        self.action_masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []

    def add(
        self,
        local_map: np.ndarray,
        scalar: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        hidden_state: np.ndarray,
    ):
        self.local_maps.append(local_map)
        self.scalars.append(scalar)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden_state)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        """Compute returns and GAE advantages."""
        advantages = []
        returns = []

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0.0  # Terminal state
            else:
                next_value = self.values[t + 1]

            if self.dones[t]:
                next_value = 0.0

            delta = (
                self.rewards[t] + gamma * next_value - self.values[t]
            )
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        self.advantages = np.array(advantages, dtype=np.float32)
        self.returns = np.array(returns, dtype=np.float32)

    def get_minibatches(self, batch_size: int):
        """Yield minibatches for training."""
        num_samples = len(self.rewards)
        indices = np.random.permutation(num_samples)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            yield (
                torch.from_numpy(np.array(self.local_maps)[batch_indices]).to(
                    self.device
                ),
                torch.from_numpy(np.array(self.scalars)[batch_indices]).to(
                    self.device
                ),
                torch.from_numpy(
                    np.array(self.action_masks)[batch_indices]
                ).to(self.device),
                torch.from_numpy(np.array(self.actions)[batch_indices]).to(
                    self.device
                ),
                torch.from_numpy(
                    np.array(self.log_probs)[batch_indices]
                ).to(self.device),
                torch.from_numpy(
                    np.array(self.returns)[batch_indices]
                ).to(self.device),
                torch.from_numpy(
                    np.array(self.advantages)[batch_indices]
                ).to(self.device),
            )


class PPOTrainer:
    """PPO trainer for Crawl DRL agent."""

    def __init__(
        self,
        config: TrainConfig,
        reward_config: RewardConfig = None,
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.config = config
        self.reward_config = reward_config or RewardConfig()
        self.device = torch.device(config.device)

        # Create directories
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.plot_dir).mkdir(parents=True, exist_ok=True)

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir)
        self.metric_logger = TrainingLogger(config.metrics_csv, config.plot_dir)

        # Model
        self.model = CrawlActorCritic(
            cnn_channels=config.cnn_channels,
            gru_hidden=config.gru_hidden,
            fc_hidden=config.fc_hidden,
            device=str(self.device),
        )

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

        # Learning rate scheduler
        if config.learning_rate_decay == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.total_timesteps // (
                    config.num_steps * config.num_envs
                ),
            )
        else:
            self.lr_scheduler = None

        # Environments
        self.envs = [
            CrawlGymnasiumEnv(opponent=config.opponent, debug=False)
            for _ in range(config.num_envs)
        ]

        # Logging
        self.global_step = 0
        self.update_count = 0
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_outcomes = deque(maxlen=100)
        self.running_episode_returns = [0.0 for _ in range(config.num_envs)]
        self.running_episode_lengths = [0 for _ in range(config.num_envs)]
        self.next_checkpoint_step = config.checkpoint_interval
        self.next_eval_step = config.eval_interval
        self.next_plot_step = config.plot_interval
        self.next_action_histogram_step = config.action_histogram_interval
        self.next_self_play_snapshot_step = config.self_play_snapshot_interval
        self.base_opponents = self._parse_opponents(config.opponent)
        self.self_play_league = SelfPlayLeague(
            pool_size=config.self_play_pool_size,
            snapshot_interval=config.self_play_snapshot_interval,
            checkpoint_dir=checkpoint_dir,
        )
        self.self_play_enabled = False

    def _parse_opponents(self, opponent) -> List[str]:
        """Normalize an opponent spec into a list for opponent sampling."""
        if isinstance(opponent, list):
            return [str(item).strip() for item in opponent if str(item).strip()]
        if isinstance(opponent, str) and "," in opponent:
            return [item.strip() for item in opponent.split(",") if item.strip()]
        return [opponent]

    def _set_env_opponents(self):
        """Push the current opponent pool into all live environments."""
        pool = self.base_opponents + self.self_play_league.list_opponents()
        opponent = pool if len(pool) > 1 else pool[0]
        self.config.opponent = opponent
        for env in self.envs:
            env.opponent = opponent
            env.current_opponent = opponent

    def maybe_update_self_play(self, win_rate: float, outcomes_count: int):
        """Add policy snapshots as opponents after the agent is reliably winning."""
        if not self.config.use_self_play:
            return

        if outcomes_count < self.config.self_play_min_episodes:
            return

        if win_rate < self.config.self_play_win_rate_threshold:
            return

        if self.global_step < self.next_self_play_snapshot_step:
            return

        snapshot_path = os.path.join(
            self.checkpoint_dir,
            f"selfplay_step_{self.global_step}.pt",
        )
        self.save_checkpoint(snapshot_path)
        self.self_play_league.add_checkpoint(snapshot_path)

        self._set_env_opponents()
        self.self_play_enabled = True
        print(
            "Self-play opponent pool updated: "
            f"{len(self.self_play_league.list_opponents())} snapshots, "
            f"{len(self.base_opponents)} base opponents"
        )
        self.next_self_play_snapshot_step = (
            self.global_step + self.config.self_play_snapshot_interval
        )

    def collect_rollouts(self) -> Tuple[RolloutBuffer, Dict[str, float]]:
        """Collect rollouts from environment."""
        buffer = RolloutBuffer(
            self.config.num_steps,
            self.config.batch_size,
            device=self.device,
        )

        obs_list = [env.reset()[0] for env in self.envs]
        hidden_states = [None] * len(self.envs)

        completed_returns = []
        completed_lengths = []
        completed_raw_rewards = []
        completed_outcomes = []
        action_counts = {action_id: 0 for action_id in ACTION_ID_TO_STRING}
        total_robot_observations = 0
        total_units_seen = 0
        valid_action_counts = []

        for step in range(self.config.num_steps):
            # Collect actions for all robots in all envs
            for env_idx, obs_dict in enumerate(obs_list):
                if not obs_dict:  # Episode done
                    obs_list[env_idx], _ = self.envs[env_idx].reset()
                    obs_dict = obs_list[env_idx]

                actions = {}
                total_units_seen += len(obs_dict)
                for uid, obs_single in obs_dict.items():
                    total_robot_observations += 1
                    valid_action_counts.append(int(np.sum(obs_single["action_mask"])))
                    local_map = torch.from_numpy(
                        obs_single["local_map"]
                    ).unsqueeze(0).to(self.device)
                    scalars = torch.from_numpy(
                        obs_single["scalars"]
                    ).unsqueeze(0).to(self.device)
                    action_mask = torch.from_numpy(
                        obs_single["action_mask"]
                    ).unsqueeze(0).to(self.device)

                    hidden = (
                        torch.from_numpy(hidden_states[env_idx]).unsqueeze(0).to(
                            self.device
                        )
                        if hidden_states[env_idx] is not None
                        else None
                    )

                    with torch.no_grad():
                        (
                            action,
                            log_prob,
                            value,
                            new_hidden,
                        ) = self.model.get_action_and_value(
                            local_map, scalars, action_mask, hidden
                        )

                        actions[uid] = action.item()
                        action_counts[actions[uid]] = action_counts.get(actions[uid], 0) + 1
                        log_prob_val = log_prob.item()
                        value_val = value.item()
                        hidden_states[env_idx] = new_hidden.squeeze(0).cpu().numpy()

                    buffer.add(
                        local_map=obs_single["local_map"],
                        scalar=obs_single["scalars"],
                        action_mask=obs_single["action_mask"],
                        action=actions[uid],
                        log_prob=log_prob_val,
                        value=value_val,
                        reward=0.0,  # Will be filled after step
                        done=False,  # Will be filled after step
                        hidden_state=hidden_states[env_idx] if hidden_states[env_idx] is not None else np.zeros(self.config.gru_hidden),
                    )

                # Step environment
                next_obs, reward, terminated, truncated, info = self.envs[
                    env_idx
                ].step(actions)
                self.running_episode_returns[env_idx] += reward
                self.running_episode_lengths[env_idx] += 1

                # Overwrite dummy rewards and dones in buffer for active robots in this step
                num_robots = len(obs_dict)
                for i in range(1, num_robots + 1):
                    buffer.rewards[-i] = reward
                    buffer.dones[-i] = terminated or truncated

                if terminated or truncated:
                    raw_reward = float(info.get("reward_raw", 0))
                    completed_returns.append(self.running_episode_returns[env_idx])
                    completed_lengths.append(self.running_episode_lengths[env_idx])
                    completed_raw_rewards.append(raw_reward)
                    if raw_reward > 0:
                        completed_outcomes.append("win")
                    elif raw_reward < 0:
                        completed_outcomes.append("loss")
                    else:
                        completed_outcomes.append("draw")

                    self.episode_rewards.append(self.running_episode_returns[env_idx])
                    self.episode_lengths.append(self.running_episode_lengths[env_idx])
                    self.episode_outcomes.append(completed_outcomes[-1])
                    self.episode_count += 1
                    self.running_episode_returns[env_idx] = 0.0
                    self.running_episode_lengths[env_idx] = 0
                    obs_list[env_idx], _ = self.envs[env_idx].reset()
                    hidden_states[env_idx] = None
                else:
                    obs_list[env_idx] = next_obs

                self.global_step += 1

        buffer.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )

        total_actions = sum(action_counts.values())
        move_north_rate = action_counts.get(1, 0) / total_actions if total_actions else 0.0
        build_rate = (
            action_counts.get(5, 0) + action_counts.get(6, 0) + action_counts.get(7, 0)
        ) / total_actions if total_actions else 0.0
        mine_transform_rate = action_counts.get(11, 0) / total_actions if total_actions else 0.0
        transfer_rate = action_counts.get(12, 0) / total_actions if total_actions else 0.0

        rollout_metrics = {
            "rollout_shaped_reward_mean": float(np.mean(buffer.rewards)) if buffer.rewards else 0.0,
            "rollout_raw_reward_mean": float(np.mean(completed_raw_rewards)) if completed_raw_rewards else 0.0,
            "rollout_completed_episodes": len(completed_returns),
            "rollout_episode_return_mean": float(np.mean(completed_returns)) if completed_returns else 0.0,
            "rollout_episode_length_mean": float(np.mean(completed_lengths)) if completed_lengths else 0.0,
            "rollout_win_rate": completed_outcomes.count("win") / len(completed_outcomes) if completed_outcomes else 0.0,
            "rollout_loss_rate": completed_outcomes.count("loss") / len(completed_outcomes) if completed_outcomes else 0.0,
            "rollout_draw_rate": completed_outcomes.count("draw") / len(completed_outcomes) if completed_outcomes else 0.0,
            "avg_units_per_step": total_units_seen / (self.config.num_steps * self.config.num_envs),
            "avg_valid_actions": float(np.mean(valid_action_counts)) if valid_action_counts else 0.0,
            "move_north_rate": move_north_rate,
            "build_rate": build_rate,
            "mine_transform_rate": mine_transform_rate,
            "transfer_rate": transfer_rate,
            "total_actions": total_actions,
            "total_robot_observations": total_robot_observations,
            "action_counts": action_counts,
        }

        return buffer, rollout_metrics

    def train_step(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform one PPO training step."""
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
        }

        num_minibatches = 0

        for epoch in range(self.config.num_epochs):
            for (
                local_maps_batch,
                scalars_batch,
                action_masks_batch,
                actions_batch,
                log_probs_batch,
                returns_batch,
                advantages_batch,
            ) in buffer.get_minibatches(self.config.batch_size):

                # Forward pass
                action_probs, values, _ = self.model(
                    local_maps_batch,
                    scalars_batch,
                    action_masks_batch,
                )

                # Policy loss (PPO)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(actions_batch)
                ratio = torch.exp(new_log_probs - log_probs_batch)
                approx_kl = (log_probs_batch - new_log_probs).mean()

                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.config.clip_range,
                        1 + self.config.clip_range,
                    )
                    * advantages_batch
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (
                    (values.squeeze(-1) - returns_batch) ** 2
                ).mean()

                # Entropy bonus
                entropy = dist.entropy().mean()
                entropy_loss = -self.config.ent_coef * entropy

                # Total loss
                total_loss = policy_loss + self.config.vf_coef * value_loss + entropy_loss

                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["total_loss"] += total_loss.item()
                metrics["approx_kl"] += approx_kl.item()

                num_minibatches += 1

        # Average metrics
        if num_minibatches > 0:
            for key in metrics:
                metrics[key] /= num_minibatches

        return metrics

    def train(self):
        """Main training loop."""
        print(
            f"Starting training for {self.config.total_timesteps:,} timesteps"
        )

        start_time = time.time()

        while self.global_step < self.config.total_timesteps:
            # Collect rollouts
            buffer, rollout_metrics = self.collect_rollouts()

            # Train on rollouts
            train_metrics = self.train_step(buffer)
            self.update_count += 1

            elapsed = time.time() - start_time
            fps = self.global_step / elapsed if elapsed > 0 else 0.0
            mean_reward = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
            mean_length = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
            outcomes = list(self.episode_outcomes)
            win_rate = outcomes.count("win") / len(outcomes) if outcomes else 0.0
            loss_rate = outcomes.count("loss") / len(outcomes) if outcomes else 0.0
            draw_rate = outcomes.count("draw") / len(outcomes) if outcomes else 0.0
            learning_rate = self.optimizer.param_groups[0]["lr"]

            eval_metrics = {}
            if self.global_step >= self.next_eval_step:
                eval_metrics = self.evaluate_current_policy()
                self.next_eval_step += self.config.eval_interval

            metrics_row = {
                "step": self.global_step,
                "updates": self.update_count,
                "elapsed_sec": elapsed,
                "fps": fps,
                "rollout_shaped_reward_mean": rollout_metrics["rollout_shaped_reward_mean"],
                "rollout_raw_reward_mean": rollout_metrics["rollout_raw_reward_mean"],
                "episode_return_mean_100": mean_reward,
                "episode_length_mean_100": mean_length,
                "win_rate_100": win_rate,
                "loss_rate_100": loss_rate,
                "draw_rate_100": draw_rate,
                "avg_units_per_step": rollout_metrics["avg_units_per_step"],
                "avg_valid_actions": rollout_metrics["avg_valid_actions"],
                "move_north_rate": rollout_metrics["move_north_rate"],
                "build_rate": rollout_metrics["build_rate"],
                "mine_transform_rate": rollout_metrics["mine_transform_rate"],
                "transfer_rate": rollout_metrics["transfer_rate"],
                "policy_loss": train_metrics["policy_loss"],
                "value_loss": train_metrics["value_loss"],
                "entropy_loss": train_metrics["entropy_loss"],
                "total_loss": train_metrics["total_loss"],
                "learning_rate": learning_rate,
                "eval_win_rate": eval_metrics.get("win_rate", ""),
                "eval_avg_reward": eval_metrics.get("avg_reward", ""),
                "eval_avg_steps": eval_metrics.get("avg_steps", ""),
            }
            self.metric_logger.log_row(metrics_row)
            self.maybe_update_self_play(win_rate, len(outcomes))

            # Logging
            if self.update_count % self.config.log_interval == 0:
                self.log_tensorboard(metrics_row, train_metrics, rollout_metrics, eval_metrics)

                print(
                    f"Step {self.global_step:,} | "
                    f"Reward100: {mean_reward:.3f} | "
                    f"Win100: {win_rate:.1%} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"Episodes: {self.episode_count} | "
                    f"FPS: {fps:.0f}"
                )

            # Checkpoint
            while self.global_step >= self.next_checkpoint_step:
                self.save_checkpoint(
                    os.path.join(
                        self.checkpoint_dir,
                        f"model_step_{self.next_checkpoint_step}.pt",
                    )
                )
                self.next_checkpoint_step += self.config.checkpoint_interval

            if self.global_step >= self.next_plot_step:
                self.metric_logger.plot()
                self.next_plot_step += self.config.plot_interval

            if self.global_step >= self.next_action_histogram_step:
                self.metric_logger.plot_action_distribution(
                    rollout_metrics["action_counts"], self.global_step
                )
                self.next_action_histogram_step += self.config.action_histogram_interval

            # Update learning rate
            if self.lr_scheduler:
                self.lr_scheduler.step()

        self.writer.close()
        self.metric_logger.plot()
        print("Training complete!")

    def log_tensorboard(
        self,
        metrics_row: Dict[str, float],
        train_metrics: Dict[str, float],
        rollout_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
    ):
        """Write rich monitoring metrics to TensorBoard."""
        step = self.global_step

        self.writer.add_scalar("rollout/shaped_reward_mean", metrics_row["rollout_shaped_reward_mean"], step)
        self.writer.add_scalar("rollout/raw_reward_mean", metrics_row["rollout_raw_reward_mean"], step)
        self.writer.add_scalar("rollout/completed_episodes", rollout_metrics["rollout_completed_episodes"], step)
        self.writer.add_scalar("episode/return_mean_100", metrics_row["episode_return_mean_100"], step)
        self.writer.add_scalar("episode/length_mean_100", metrics_row["episode_length_mean_100"], step)
        self.writer.add_scalar("outcome/win_rate_100", metrics_row["win_rate_100"], step)
        self.writer.add_scalar("outcome/loss_rate_100", metrics_row["loss_rate_100"], step)
        self.writer.add_scalar("outcome/draw_rate_100", metrics_row["draw_rate_100"], step)

        self.writer.add_scalar("train/policy_loss", train_metrics["policy_loss"], step)
        self.writer.add_scalar("train/value_loss", train_metrics["value_loss"], step)
        self.writer.add_scalar("train/entropy_loss", train_metrics["entropy_loss"], step)
        self.writer.add_scalar("train/total_loss", train_metrics["total_loss"], step)
        self.writer.add_scalar("train/approx_kl", train_metrics["approx_kl"], step)
        self.writer.add_scalar("train/learning_rate", metrics_row["learning_rate"], step)

        self.writer.add_scalar("runtime/fps", metrics_row["fps"], step)
        self.writer.add_scalar("runtime/elapsed_sec", metrics_row["elapsed_sec"], step)
        self.writer.add_scalar("game/avg_units_per_step", metrics_row["avg_units_per_step"], step)
        self.writer.add_scalar("game/avg_valid_actions", metrics_row["avg_valid_actions"], step)

        self.writer.add_scalar("strategy/move_north_rate", metrics_row["move_north_rate"], step)
        self.writer.add_scalar("strategy/build_rate", metrics_row["build_rate"], step)
        self.writer.add_scalar("strategy/mine_transform_rate", metrics_row["mine_transform_rate"], step)
        self.writer.add_scalar("strategy/transfer_rate", metrics_row["transfer_rate"], step)

        action_counts = rollout_metrics["action_counts"]
        for action_id, count in action_counts.items():
            action_name = ACTION_ID_TO_STRING.get(action_id, str(action_id)).lower()
            self.writer.add_scalar(f"actions/{action_name}", count, step)

        if eval_metrics:
            self.writer.add_scalar("eval/win_rate", eval_metrics["win_rate"], step)
            self.writer.add_scalar("eval/avg_reward", eval_metrics["avg_reward"], step)
            self.writer.add_scalar("eval/avg_steps", eval_metrics["avg_steps"], step)
            self.writer.add_scalar("eval/wins", eval_metrics["wins"], step)
            self.writer.add_scalar("eval/losses", eval_metrics["losses"], step)
            self.writer.add_scalar("eval/draws", eval_metrics["draws"], step)

        self.writer.flush()

    def evaluate_current_policy(self) -> Dict[str, float]:
        """Run deterministic evaluation episodes and return aggregate metrics."""
        print(
            f"Evaluating current policy for {self.config.eval_episodes} episodes "
            f"vs {self.config.opponent}..."
        )
        was_training = self.model.training
        evaluator = Evaluator(self.model, device=str(self.device))
        results = evaluator.evaluate(
            opponent=self.config.opponent,
            num_episodes=self.config.eval_episodes,
            deterministic=True,
        )
        if was_training:
            self.model.train()
        print(
            f"Eval | Win: {results['win_rate']:.1%} | "
            f"W/L/D: {results['wins']}/{results['losses']}/{results['draws']} | "
            f"Reward: {results['avg_reward']:.3f} | Steps: {results['avg_steps']:.1f}"
        )
        return results

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "update_count": self.update_count,
                "episode_count": self.episode_count,
                "self_play_pool": self.self_play_league.list_opponents(),
                "self_play_enabled": self.self_play_enabled,
                "config": vars(self.config),
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.update_count = checkpoint.get("update_count", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.self_play_league.pool = checkpoint.get("self_play_pool", [])
        self.self_play_enabled = checkpoint.get(
            "self_play_enabled", bool(self.self_play_league.pool)
        )
        if self.self_play_league.pool:
            self._set_env_opponents()
        self.next_checkpoint_step = (
            (self.global_step // self.config.checkpoint_interval) + 1
        ) * self.config.checkpoint_interval
        self.next_eval_step = (
            (self.global_step // self.config.eval_interval) + 1
        ) * self.config.eval_interval
        self.next_plot_step = (
            (self.global_step // self.config.plot_interval) + 1
        ) * self.config.plot_interval
        self.next_action_histogram_step = (
            (self.global_step // self.config.action_histogram_interval) + 1
        ) * self.config.action_histogram_interval
        self.next_self_play_snapshot_step = (
            (self.global_step // self.config.self_play_snapshot_interval) + 1
        ) * self.config.self_play_snapshot_interval
        print(f"Loaded checkpoint from {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total timesteps to train",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help="Opponent agent: random, .pt/.py path, or comma-separated pool",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2048,
        help="Rollout steps collected per environment before each PPO update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO minibatch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10_000,
        help="Save a model checkpoint every N environment steps",
    )
    parser.add_argument(
        "--use-self-play",
        action="store_true",
        help="Add current-policy snapshots to the opponent pool after it is reliably winning",
    )
    parser.add_argument(
        "--self-play-snapshot-interval",
        type=int,
        default=100_000,
        help="Save a self-play opponent snapshot every N steps once the win-rate gate is met",
    )
    parser.add_argument(
        "--self-play-pool-size",
        type=int,
        default=10,
        help="Maximum number of self-play snapshots to sample as opponents",
    )
    parser.add_argument(
        "--self-play-win-rate-threshold",
        type=float,
        default=0.90,
        help="Rolling Win100 threshold required before adding self-play opponents",
    )
    parser.add_argument(
        "--self-play-min-episodes",
        type=int,
        default=100,
        help="Minimum completed rolling episodes before the self-play win-rate gate can open",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50_000,
        help="Run evaluation every N environment steps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes per evaluation",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Write TensorBoard console metrics every N PPO updates",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="TensorBoard and CSV log directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--plot-interval",
        type=int,
        default=50_000,
        help="Regenerate PNG training plots every N environment steps",
    )
    parser.add_argument(
        "--action-histogram-interval",
        type=int,
        default=10_000,
        help="Regenerate action distribution PNG every N environment steps",
    )
    args = parser.parse_args()

    config = TrainConfig(
        total_timesteps=args.total_timesteps,
        opponent=args.opponent,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        use_self_play=args.use_self_play,
        self_play_snapshot_interval=args.self_play_snapshot_interval,
        self_play_pool_size=args.self_play_pool_size,
        self_play_win_rate_threshold=args.self_play_win_rate_threshold,
        self_play_min_episodes=args.self_play_min_episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        log_interval=args.log_interval,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        metrics_csv=os.path.join(args.log_dir, "training_metrics.csv"),
        plot_dir=os.path.join(args.log_dir, "plots"),
        plot_interval=args.plot_interval,
        action_histogram_interval=args.action_histogram_interval,
    )

    trainer = PPOTrainer(config, log_dir=config.log_dir, checkpoint_dir=config.checkpoint_dir)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()


if __name__ == "__main__":
    main()
