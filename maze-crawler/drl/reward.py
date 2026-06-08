"""Reward shaping utilities for DRL agent."""

import numpy as np
from typing import Dict, Tuple


class RewardShaper:
    """Computes shaped rewards for Crawl gameplay."""

    def __init__(self, config):
        self.config = config
        self.prev_state = {}  # Track state changes for delta rewards
        self.idle_counts = {}

        # Reward coefficients (can be tuned)
        self.coefficients = {
            "win": 10.0,
            "loss": -10.0,
            "draw": 0.0,
            "crystal": 1.0,  # Multiply by (value / 50)
            "mine_income": 0.1,  # Per mine per turn
            "mine_built": 3.0,
            "exploration": 0.02,  # Per new cell
            "crush_win": 1.5,
            "robot_loss": -0.5,
            "friendly_fire": -3.0,
            "survival": 0.01,  # Per turn factory alive
            "scroll_danger": -0.1,  # Per robot near southBound
            "scroll_death": -5.0,
            "energy_delta": 0.001,  # 0.001 * delta_energy per turn
            "idle_penalty": -0.05,  # Per robot idle >5 turns
        }

    def compute_reward(
        self, obs, prev_obs, done: bool, reward: float, player_idx: int
    ) -> float:
        """
        Compute shaped reward for a single step.

        Args:
            obs: Current observation
            prev_obs: Previous observation
            done: Whether episode is done
            reward: Base Kaggle reward (win/loss/draw)
            player_idx: Player index (0 or 1)

        Returns:
            Shaped reward (scalar)
        """
        shaped = 0.0

        # === Terminal Rewards ===
        if done:
            if reward > 0:
                shaped += self.coefficients["win"]
            elif reward < 0:
                shaped += self.coefficients["loss"]
            else:
                shaped += self.coefficients["draw"]
            return float(shaped)  # Terminal rewards stay at +/-10 per plan.

        # === Per-turn shaping (only if not terminal) ===

        # Get my robots
        my_robots = {
            uid: data
            for uid, data in obs.robots.items()
            if data[4] == player_idx
        }

        # === Mine Income ===
        mine_energy_earned = 0.0
        for mine_key, mine_data in obs.mines.items():
            col, row = map(int, mine_key.split(","))
            owner = mine_data[2]
            if owner == player_idx:
                # Each robot standing on mine collects 50 energy
                for uid, data in my_robots.items():
                    if data[1] == col and data[2] == row:
                        mine_energy_earned += 50.0

        if mine_energy_earned > 0:
            shaped += self.coefficients["mine_income"] * (
                min(mine_energy_earned, 150) / 50
            )

        # === Crystal collection and mine construction ===
        if prev_obs:
            shaped += self.coefficients["crystal"] * self._crystal_reward(
                obs, prev_obs, player_idx
            )
            shaped += self.coefficients["mine_built"] * self._new_owned_mines(
                obs, prev_obs, player_idx
            )

        # === Exploration (new cells discovered) ===
        if prev_obs:
            new_cells = 0
            for idx, wall in enumerate(obs.walls):
                if wall != -1:  # Discovered
                    prev_wall = prev_obs.walls[idx] if idx < len(prev_obs.walls) else -1
                    if prev_wall == -1:  # Was undiscovered
                        new_cells += 1

            if new_cells > 0:
                shaped += self.coefficients["exploration"] * new_cells

        # === Survival ===
        if my_robots:
            shaped += self.coefficients["survival"]

        # === Robot Loss (from combat or scroll) ===
        if prev_obs:
            prev_my_robots = {
                uid: data
                for uid, data in prev_obs.robots.items()
                if data[4] == player_idx
            }
            robot_loss_count = len(prev_my_robots) - len(my_robots)
            if robot_loss_count > 0:
                shaped += self.coefficients["robot_loss"] * robot_loss_count

            crush_wins = self._crush_wins(obs, prev_obs, player_idx)
            if crush_wins > 0:
                shaped += self.coefficients["crush_win"] * crush_wins

            friendly_fire = self._friendly_fire_count(obs, prev_obs, player_idx)
            if friendly_fire > 0:
                shaped += self.coefficients["friendly_fire"] * friendly_fire

            scroll_deaths = self._scroll_deaths(obs, prev_obs, player_idx)
            if scroll_deaths > 0:
                shaped += self.coefficients["scroll_death"] * scroll_deaths

        # === Scroll Danger ===
        robots_near_south = 0
        for uid, data in my_robots.items():
            row = data[2]
            if row - obs.southBound < 3:
                robots_near_south += 1

        if robots_near_south > 0:
            shaped += self.coefficients["scroll_danger"] * robots_near_south

        # === Energy Delta ===
        total_energy = sum(data[3] for data in my_robots.values())
        if hasattr(self, "_prev_total_energy"):
            energy_delta = total_energy - self._prev_total_energy
            shaped += self.coefficients["energy_delta"] * energy_delta
        self._prev_total_energy = total_energy

        # === Anti-stagnation: penalize robots that stay in place for too long ===
        if prev_obs:
            idle_penalty_count = self._update_idle_counts(obs, prev_obs, my_robots)
            if idle_penalty_count > 0:
                shaped += self.coefficients["idle_penalty"] * idle_penalty_count

        # Clip to [-1, 1] per step (except terminal)
        return float(np.clip(shaped, -1.0, 1.0))

    def _crystal_reward(self, obs, prev_obs, player_idx: int) -> float:
        reward = 0.0
        my_positions = {
            (data[1], data[2]) for data in obs.robots.values() if data[4] == player_idx
        }
        for key, prev_value in prev_obs.crystals.items():
            cur_value = obs.crystals.get(key, 0)
            if cur_value < prev_value:
                col, row = map(int, key.split(","))
                if (col, row) in my_positions:
                    reward += min((prev_value - cur_value) / 50.0, 1.0)
        return reward

    def _new_owned_mines(self, obs, prev_obs, player_idx: int) -> int:
        count = 0
        for key, mine_data in obs.mines.items():
            if mine_data[2] != player_idx:
                continue
            prev_mine = prev_obs.mines.get(key)
            if prev_mine is None or prev_mine[2] != player_idx:
                count += 1
        return count

    def _crush_wins(self, obs, prev_obs, player_idx: int) -> int:
        enemy_idx = 1 - player_idx
        prev_enemies = {
            uid: data for uid, data in prev_obs.robots.items() if data[4] == enemy_idx
        }
        cur_enemy_uids = {uid for uid, data in obs.robots.items() if data[4] == enemy_idx}
        my_positions = {
            (data[1], data[2]) for data in obs.robots.values() if data[4] == player_idx
        }
        return sum(
            1
            for uid, data in prev_enemies.items()
            if uid not in cur_enemy_uids and (data[1], data[2]) in my_positions
        )

    def _friendly_fire_count(self, obs, prev_obs, player_idx: int) -> int:
        prev_my = {
            uid: data for uid, data in prev_obs.robots.items() if data[4] == player_idx
        }
        cur_uids = set(obs.robots.keys())
        cur_positions = {
            (data[1], data[2]) for data in obs.robots.values() if data[4] == player_idx
        }
        return sum(
            1
            for uid, data in prev_my.items()
            if uid not in cur_uids and (data[1], data[2]) in cur_positions
        )

    def _scroll_deaths(self, obs, prev_obs, player_idx: int) -> int:
        prev_my = {
            uid: data for uid, data in prev_obs.robots.items() if data[4] == player_idx
        }
        cur_uids = set(obs.robots.keys())
        return sum(
            1
            for uid, data in prev_my.items()
            if uid not in cur_uids and data[2] < obs.southBound
        )

    def _update_idle_counts(self, obs, prev_obs, my_robots: Dict[str, list]) -> int:
        prev_positions = {
            uid: (data[1], data[2])
            for uid, data in prev_obs.robots.items()
            if uid in my_robots
        }
        live_uids = set(my_robots.keys())
        for dead_uid in list(self.idle_counts.keys()):
            if dead_uid not in live_uids:
                del self.idle_counts[dead_uid]

        penalized = 0
        for uid, data in my_robots.items():
            cur_pos = (data[1], data[2])
            if prev_positions.get(uid) == cur_pos:
                self.idle_counts[uid] = self.idle_counts.get(uid, 0) + 1
            else:
                self.idle_counts[uid] = 0
            if self.idle_counts[uid] > 5:
                penalized += 1
        return penalized

    def set_coefficients(self, coefficients: Dict[str, float]):
        """Update reward coefficients."""
        self.coefficients.update(coefficients)


if __name__ == "__main__":
    print("reward.py loaded successfully")
