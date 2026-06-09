"""Gymnasium wrapper for Kaggle Crawl environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from kaggle_environments import make, evaluate
from typing import Tuple, Dict, Any
import os
import sys

from .obs_utils import extract_local_view, extract_scalars
from .action_utils import (
    compute_action_mask,
    resolve_action_direction_heuristic,
    ACTION_ID_TO_STRING,
)
from .reward import RewardShaper


class CrawlGymnasiumEnv(gym.Env):
    """
    Gymnasium wrapper for Crawl environment.

    This wrapper treats each robot as an independent agent with decentralized control.
    Multi-agent handling via parameter sharing:
    - Each step returns observation & mask for all alive robots
    - Takes action dict indexed by robot UID
    - Handles variable robot counts naturally
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent: str = "random",
        scroll_speed: float = 1.0,
        fog_enabled: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            opponent: Opponent agent module path, "random", or path to a PyTorch .pt model
            scroll_speed: Speed multiplier (0.0 to 1.0)
            fog_enabled: Whether fog-of-war is active
            debug: Enable debug output
        """
        super().__init__()

        self.opponent = opponent
        self.scroll_speed = scroll_speed
        self.fog_enabled = fog_enabled
        self.debug = debug

        # Map scroll_speed curriculum to env configuration
        custom_config = {}
        if scroll_speed == 0:
            custom_config["scrollStartInterval"] = 999999
            custom_config["scrollEndInterval"] = 999999
        elif scroll_speed <= 0.25:
            custom_config["scrollStartInterval"] = 4
            custom_config["scrollEndInterval"] = 4

        self.env = make("crawl", configuration=custom_config, debug=debug)
        self.config = self.env.configuration
        self.reward_shaper = RewardShaper(self.config)

        # Action space per robot
        self.action_space = spaces.Discrete(13)

        # Observation space per robot
        # Local map: 12 channels, 11×11
        # Scalars: 13 features
        self.observation_space = spaces.Dict(
            {
                "local_map": spaces.Box(
                    low=-1.0, high=1.0, shape=(12, 11, 11), dtype=np.float32
                ),
                "scalars": spaces.Box(
                    low=-1.0, high=1.0, shape=(13,), dtype=np.float32
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(13,), dtype=np.uint8),
            }
        )

        self.steps = 0
        self.max_steps = 500
        self.prev_obs = None
        self.current_opponent = opponent

    def reset(self) -> Tuple[Dict, Dict]:
        """Reset environment and return observation."""
        self.steps = 0
        self.prev_obs = None

        # Choose opponent for this episode if multiple are provided
        if isinstance(self.opponent, list):
            self.current_opponent = np.random.choice(self.opponent)
        elif isinstance(self.opponent, str) and "," in self.opponent:
            self.current_opponent = np.random.choice(self.opponent.split(","))
        else:
            self.current_opponent = self.opponent

        # Run initial step to get observation
        state = self.env.reset()
        self.state = state

        obs = self._get_obs()
        info = {"step": 0}

        return obs, info

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            actions: {uid: action_id} for player 0's robots only

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.steps += 1

        # Convert action IDs to action strings
        agent_actions = {}
        if self.state and len(self.state) > 0:
            player_obs = self.state[0].observation
            my_robots = {
                uid: data
                for uid, data in player_obs.robots.items()
                if data[4] == 0
            }

            for uid in my_robots.keys():
                if uid in actions:
                    action_id = actions[uid]
                    robot_data = my_robots[uid]
                    action_str = self._resolve_action(
                        action_id, robot_data, player_obs
                    )
                    agent_actions[uid] = action_str
                else:
                    agent_actions[uid] = "IDLE"

        # Get opponent actions
        if self.state and len(self.state) > 1:
            player_obs = self.state[0].observation
            opponent_obs = self.state[1].observation

            # Copy global environment fields to opponent observation
            for field in ['southBound', 'northBound', 'step', 'globalWalls', 'globalCrystals', 'globalRobots', 'globalMines', 'globalMiningNodes']:
                if hasattr(player_obs, field):
                    setattr(opponent_obs, field, getattr(player_obs, field))

            if isinstance(self.current_opponent, str) and self.current_opponent == "random":
                opponent_actions = self._random_actions(opponent_obs)
            else:
                opponent_actions = self._get_opponent_actions(opponent_obs)
        else:
            opponent_actions = {}

        # Convert to list format for env.step
        action_list = [agent_actions, opponent_actions]

        # Step environment
        self.state = self.env.step(action_list)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        if self.state and len(self.state) > 0:
            reward_value = self.state[0].reward if hasattr(
                self.state[0], "reward"
            ) else 0
            done = self.state[0].status != "ACTIVE" if hasattr(
                self.state[0], "status"
            ) else False
        else:
            reward_value = 0
            done = False

        # Use reward shaper
        shaped_reward = self.reward_shaper.compute_reward(
            self.state[0].observation if self.state and len(self.state) > 0 else None,
            self.prev_obs,
            done,
            reward_value,
            0,  # player 0
        )

        self.prev_obs = (
            self.state[0].observation if self.state and len(self.state) > 0 else None
        )

        truncated = self.steps >= self.max_steps
        terminated = done

        info = {
            "step": self.steps,
            "reward_raw": reward_value,
            "terminated": terminated,
        }

        return obs, shaped_reward, terminated, truncated, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Extract observation for all alive robots (player 0)."""
        if not self.state or len(self.state) < 1:
            return {}

        player_obs = self.state[0].observation
        obs_dict = {}

        my_robots = {
            uid: data
            for uid, data in player_obs.robots.items()
            if data[4] == 0  # Player 0
        }

        for uid, robot_data in my_robots.items():
            col, row = robot_data[1], robot_data[2]

            local_map = extract_local_view(
                player_obs, col, row, self.config, radius=5, player_idx=0, fog_enabled=self.fog_enabled
            )
            scalars = extract_scalars(robot_data, player_obs, self.config)
            mask = compute_action_mask(robot_data, player_obs, self.config)

            obs_dict[uid] = {
                "local_map": local_map,
                "scalars": scalars,
                "action_mask": mask.astype(np.uint8),
            }

        return obs_dict

    def _resolve_action(
        self, action_id: int, robot_data: list, obs
    ) -> str:
        """Convert action ID to action string, resolving directions."""
        if action_id in [9, 10, 12, 8]:  # Directional actions
            return resolve_action_direction_heuristic(
                action_id, robot_data, obs, self.config
            )
        else:
            return ACTION_ID_TO_STRING.get(action_id, "IDLE")

    def _random_actions(self, obs) -> Dict[str, str]:
        """Generate random valid actions for opponent."""
        my_robots = {
            uid: data
            for uid, data in obs.robots.items()
            if data[4] == 1  # Player 1 (opponent)
        }

        actions = {}
        for uid, robot_data in my_robots.items():
            mask = compute_action_mask(robot_data, obs, self.config)
            valid_actions = [i for i, v in enumerate(mask) if v]
            if valid_actions:
                action_id = np.random.choice(valid_actions)
                actions[uid] = self._resolve_action(action_id, robot_data, obs)
            else:
                actions[uid] = "IDLE"

        return actions

    def _load_opponent_model(self):
        """Lazily load PyTorch neural network model for opponent agent."""
        if not hasattr(self, "_opp_model") or self._opp_model_path != self.current_opponent:
            import torch
            from .model import CrawlActorCritic
            self._opp_model = CrawlActorCritic(device="cpu")
            checkpoint = torch.load(self.current_opponent, map_location="cpu")
            self._opp_model.load_state_dict(checkpoint["model_state_dict"])
            self._opp_model.eval()
            self._opp_model_path = self.current_opponent
            self._opp_hidden_states = {}

    def _opp_model_actions(self, obs) -> Dict[str, str]:
        """Execute action selection via neural network model for opponent."""
        self._load_opponent_model()
        import torch

        my_robots = {
            uid: data
            for uid, data in obs.robots.items()
            if data[4] == 1  # Player 1 (opponent)
        }

        # Clean up hidden states for dead robots
        live_uids = set(my_robots.keys())
        for dead_uid in list(self._opp_hidden_states.keys()):
            if dead_uid not in live_uids:
                del self._opp_hidden_states[dead_uid]

        actions = {}
        for uid, robot_data in my_robots.items():
            col, row = robot_data[1], robot_data[2]

            # Extract local view for player 1, respect fog curriculum settings
            local_map = extract_local_view(
                obs, col, row, self.config, radius=5, player_idx=1, fog_enabled=self.fog_enabled
            )
            scalars = extract_scalars(robot_data, obs, self.config)
            mask = compute_action_mask(robot_data, obs, self.config)

            hidden = (
                torch.from_numpy(self._opp_hidden_states[uid]).unsqueeze(0)
                if uid in self._opp_hidden_states
                else None
            )

            with torch.no_grad():
                local_map_t = torch.from_numpy(local_map).unsqueeze(0)
                scalars_t = torch.from_numpy(scalars).unsqueeze(0)
                mask_t = torch.from_numpy(mask).unsqueeze(0)

                probs, _, new_hidden = self._opp_model(local_map_t, scalars_t, mask_t, hidden)
                action_id = probs.argmax(dim=-1).item()
                self._opp_hidden_states[uid] = new_hidden.squeeze(0).numpy()

            actions[uid] = self._resolve_action(action_id, robot_data, obs)

        return actions

    def _get_opponent_actions(self, obs) -> Dict[str, str]:
        """Get opponent actions from opponent agent."""
        try:
            if isinstance(self.current_opponent, str) and self.current_opponent.endswith(".pt"):
                return self._opp_model_actions(obs)
            elif isinstance(self.current_opponent, str) and self.current_opponent.endswith(".py"):
                # Import agent from file
                module_path = self.current_opponent
                spec = __import__("importlib.util").util.spec_from_file_location(
                    "opponent_agent", module_path
                )
                module = __import__("importlib.util").util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, "agent"):
                    actions = module.agent(obs, self.config)
                    return actions
        except Exception as e:
            print(f"Error loading opponent agent {self.current_opponent}: {e}")

        # Fallback to random
        return self._random_actions(obs)

    def render(self, mode: str = "human"):
        """Render environment (not fully implemented)."""
        pass

    def close(self):
        """Close environment."""
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    print("env_wrapper.py loaded successfully")
    # Test
    env = CrawlGymnasiumEnv(opponent="random", debug=False)
    obs, info = env.reset()
    print(f"Reset successful. Robots: {list(obs.keys())}")

    for _ in range(10):
        if obs:
            actions = {uid: 0 for uid in obs.keys()}  # IDLE
            obs, reward, done, truncated, info = env.step(actions)
            print(f"Step {info['step']}: reward={reward:.3f}, done={done}")
            if done or truncated:
                break
