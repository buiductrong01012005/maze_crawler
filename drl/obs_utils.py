"""Observation extraction utilities for DRL agent.

Converts raw Kaggle Crawl observations into neural network-ready tensors.
"""

import numpy as np
from typing import Tuple, Dict


def extract_local_view(
    obs, robot_col: int, robot_row: int, config, radius: int = 5, player_idx: int = 0, fog_enabled: bool = True
) -> np.ndarray:
    """
    Extract 11×11 local egocentric view around a robot.

    Args:
        obs: Kaggle observation object
        robot_col, robot_row: Robot position
        config: Kaggle config object
        radius: Local view radius (default 5 = 11×11)
        player_idx: Player index (0 or 1)
        fog_enabled: If False, uses global environment info (no-fog curriculum)

    Returns:
        channels: (12, 2*radius+1, 2*radius+1) float32 array
    """
    width = config.width
    local_size = 2 * radius + 1
    channels = np.zeros((12, local_size, local_size), dtype=np.float32)

    # Robot type max energies
    type_max_energy = {
        0: 1000,  # Factory (unlimited, cap at 1000)
        1: 100,   # Scout
        2: 300,   # Worker
        3: 500,   # Miner
    }

    # Map inputs to global or local sources depending on fog_enabled
    use_global = not fog_enabled and hasattr(obs, "globalWalls")
    walls_source = obs.globalWalls if use_global else obs.walls
    robots_source = obs.globalRobots if use_global else obs.robots
    crystals_source = obs.globalCrystals if use_global else obs.crystals
    mining_nodes_source = obs.globalMiningNodes if use_global else obs.miningNodes
    mines_source = obs.globalMines if use_global else obs.mines

    my_robots_dict = {}
    enemy_robots_dict = {}
    for uid, data in robots_source.items():
        rtype, col, row, energy, owner = data[0], data[1], data[2], data[3], data[4]
        if owner == player_idx:
            my_robots_dict[(col, row)] = (rtype, energy)
        else:
            enemy_robots_dict[(col, row)] = (rtype, energy)

    mines_dict = {}
    for mine_key, mine_data in mines_source.items():
        col, row = map(int, mine_key.split(","))
        mines_dict[(col, row)] = mine_data

    mining_nodes_dict = set(
        (int(k.split(",")[0]), int(k.split(",")[1])) for k in mining_nodes_source
    )

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r, c = robot_row + dy, robot_col + dx
            ly, lx = dy + radius, dx + radius

            # Out of map or out of northern/southern boundary
            if c < 0 or c >= width or r < obs.southBound or r >= obs.northBound:
                continue

            idx = (r - obs.southBound) * width + c
            if idx < 0 or idx >= len(walls_source):
                continue

            wall = walls_source[idx]

            # === Fog of War ===
            if wall == -1:  # Unexplored
                channels[4, ly, lx] = 0  # visibility = 0
                continue

            # === Already explored ===
            channels[4, ly, lx] = 1
            channels[0, ly, lx] = 1 if (wall & 1) else 0  # North wall
            channels[1, ly, lx] = 1 if (wall & 2) else 0  # East wall
            channels[2, ly, lx] = 1 if (wall & 4) else 0  # South wall
            channels[3, ly, lx] = 1 if (wall & 8) else 0  # West wall

            # === My robots ===
            if (c, r) in my_robots_dict:
                rtype, energy = my_robots_dict[(c, r)]
                channels[5, ly, lx] = rtype
                max_en = type_max_energy.get(rtype, 1000)
                channels[6, ly, lx] = min(energy / max_en, 1.0)

            # === Enemy robots ===
            if (c, r) in enemy_robots_dict:
                rtype, energy = enemy_robots_dict[(c, r)]
                channels[7, ly, lx] = rtype

            # === Crystals (visible only) ===
            crystal_key = f"{c},{r}"
            if crystal_key in crystals_source:
                channels[8, ly, lx] = min(crystals_source[crystal_key] / 50.0, 1.0)

            # === Mining nodes (visible only) ===
            if (c, r) in mining_nodes_dict:
                channels[9, ly, lx] = 1

            # === Mines (remembered) ===
            if (c, r) in mines_dict:
                mine_data = mines_dict[(c, r)]
                owner = mine_data[2]
                channels[10, ly, lx] = 1 if owner == player_idx else -1

            # === Scroll danger ===
            viewport_height = obs.northBound - obs.southBound
            if viewport_height > 0:
                danger = 1.0 - (r - obs.southBound) / viewport_height
                channels[11, ly, lx] = np.clip(danger, 0.0, 1.0)

    return channels


def extract_scalars(robot_data: list, obs, config) -> np.ndarray:
    """
    Extract scalar features for a single robot.

    Args:
        robot_data: [type, col, row, energy, owner, move_cd, jump_cd, build_cd]
        obs: Kaggle observation
        config: Kaggle config

    Returns:
        scalars: (13,) float32 array

    Features:
        0-3: robot_type one-hot (factory/scout/worker/miner)
        4: energy_ratio (0.0-1.0)
        5: move_cooldown (0.0-1.0)
        6: build_cooldown (0.0-1.0)  # only factory
        7: jump_cooldown (0.0-1.0)   # only factory
        8: distance_to_south (0.0-1.0)
        9: game_progress (0.0-1.0)
        10: scroll_speed (0.0-1.0)
        11: total_team_energy (0.0-1.0)
        12: total_team_units (0.0-1.0)
    """
    rtype, col, row, energy, owner, move_cd, jump_cd, build_cd = robot_data

    type_max_energy = {0: 1000, 1: 100, 2: 300, 3: 500}
    move_period = {0: 2, 1: 1, 2: 2, 3: 2}

    scalars = np.zeros(13, dtype=np.float32)

    # One-hot robot type
    scalars[rtype] = 1.0  # indices 0-3

    # Energy ratio
    max_en = type_max_energy.get(rtype, 1000)
    scalars[4] = min(energy / max_en, 1.0)

    # Cooldowns (normalized by their max)
    scalars[5] = move_cd / move_period.get(rtype, 1)
    scalars[6] = build_cd / 10.0  # factory build cooldown max = 10
    scalars[7] = jump_cd / 20.0  # factory jump cooldown max = 20

    # Distance to south boundary
    viewport_height = obs.northBound - obs.southBound
    if viewport_height > 0:
        scalars[8] = (row - obs.southBound) / viewport_height
    else:
        scalars[8] = 0.5

    # Game progress (0 to 500 steps)
    scalars[9] = obs.step / 500.0 if hasattr(obs, "step") else 0.0

    # Scroll speed (0.25 at start, 1.0 at step 400+)
    step = obs.step if hasattr(obs, "step") else 0
    if step < 400:
        scroll_speed = 0.25 + (0.75 * step / 400)
    else:
        scroll_speed = 1.0
    scalars[10] = scroll_speed

    # Total team energy and units
    team_energy = 0.0
    team_units = 0
    for uid, data in obs.robots.items():
        if data[4] == owner:  # Same player
            team_energy += data[3]
            team_units += 1

    scalars[11] = min(team_energy / 5000.0, 1.0)
    scalars[12] = min(team_units / 20.0, 1.0)

    return scalars


# Test functions
if __name__ == "__main__":
    print("obs_utils.py loaded successfully")
