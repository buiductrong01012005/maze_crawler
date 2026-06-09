"""Evaluation utilities for DRL agent."""

import torch
import numpy as np
from typing import Dict, List, Tuple
from .env_wrapper import CrawlGymnasiumEnv
from .model import CrawlActorCritic
from .obs_utils import extract_local_view, extract_scalars
from .action_utils import compute_action_mask


class Evaluator:
    """Evaluates trained model against opponents."""

    def __init__(
        self,
        model: CrawlActorCritic,
        device: str = "cuda",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.eval()

    def evaluate(
        self,
        opponent: str = "random",
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent against opponent.

        Args:
            opponent: "random" or path to agent module
            num_episodes: Number of evaluation episodes
            deterministic: Use argmax instead of sampling

        Returns:
            Dictionary with metrics
        """
        wins = 0
        losses = 0
        draws = 0
        total_reward = 0.0
        total_steps = 0

        for episode in range(num_episodes):
            env = CrawlGymnasiumEnv(opponent=opponent, debug=False)
            try:
                obs, _ = env.reset()
                hidden_states = {}
                done = False
                info = {"reward_raw": 0}
                episode_reward = 0.0
                steps = 0

                while not done and steps < 500:
                    if not obs:
                        break

                    actions = {}
                    for uid, obs_single in obs.items():
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
                            torch.from_numpy(
                                hidden_states.get(uid)
                            ).unsqueeze(0).to(self.device)
                            if uid in hidden_states
                            else None
                        )

                        with torch.no_grad():
                            action_probs, value, new_hidden = self.model(
                                local_map, scalars, action_mask, hidden
                            )

                            if deterministic:
                                action = action_probs.argmax(dim=-1).item()
                            else:
                                dist = torch.distributions.Categorical(
                                    action_probs
                                )
                                action = dist.sample().item()

                            actions[uid] = action
                            hidden_states[uid] = (
                                new_hidden.squeeze(0).cpu().numpy()
                            )

                    obs, reward, done, truncated, info = env.step(actions)
                    episode_reward += reward
                    steps += 1

                total_reward += episode_reward
                total_steps += steps

                if info.get("reward_raw", 0) > 0:
                    wins += 1
                elif info.get("reward_raw", 0) < 0:
                    losses += 1
                else:
                    draws += 1
            finally:
                env.close()

        win_rate = wins / num_episodes if num_episodes > 0 else 0.0
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        avg_steps = total_steps / num_episodes if num_episodes > 0 else 0.0

        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "num_episodes": num_episodes,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--opponent", type=str, default="random", help="Opponent agent"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Number of episodes"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = CrawlActorCritic(device=str(device))

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model, device=str(device))
    results = evaluator.evaluate(
        opponent=args.opponent,
        num_episodes=args.num_episodes,
        deterministic=True,
    )

    print(f"Win rate vs {args.opponent}: {results['win_rate']:.1%}")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print(f"Avg reward: {results['avg_reward']:.3f}")
    print(f"Avg steps: {results['avg_steps']:.1f}")
