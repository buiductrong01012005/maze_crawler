"""Action space utilities for DRL agent.

Handles 13 unified actions with type-specific masking.
"""

import numpy as np


# Action ID to string mapping (unified across all robot types)
ACTION_ID_TO_STRING = {
    0: "IDLE",
    1: "NORTH",
    2: "SOUTH",
    3: "EAST",
    4: "WEST",
    5: "BUILD_SCOUT",
    6: "BUILD_WORKER",
    7: "BUILD_MINER",
    8: "JUMP_NORTH",
    9: "REMOVE_WALL",
    10: "BUILD_WALL",
    11: "TRANSFORM",
    12: "TRANSFER",
}

STRING_TO_ACTION_ID = {v: k for k, v in ACTION_ID_TO_STRING.items()}


DIRECTION_DELTAS = {
    1: (0, 1),   # NORTH
    2: (0, -1),  # SOUTH
    3: (1, 0),   # EAST
    4: (-1, 0),  # WEST
}

DIRECTION_BITS = {1: 1, 2: 4, 3: 2, 4: 8}


def _get_field(obj, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        return obj[name]
    except (KeyError, TypeError):
        return default


def _in_bounds(col: int, row: int, obs, config) -> bool:
    width = _get_field(config, "width", 20)
    south_bound = _get_field(obs, "southBound", 0)
    north_bound = _get_field(obs, "northBound", _get_field(config, "height", width))
    return 0 <= col < width and south_bound <= row < north_bound


def _wall_at(col: int, row: int, obs, config, default: int = 15) -> int:
    if not _in_bounds(col, row, obs, config):
        return default

    width = _get_field(config, "width", 20)
    south_bound = _get_field(obs, "southBound", 0)
    walls = _get_field(obs, "walls", [])
    idx = (row - south_bound) * width + col
    if 0 <= idx < len(walls) and walls[idx] != -1:
        return int(walls[idx])
    return default


def _iter_adjacent(col: int, row: int):
    yield 1, col, row + 1
    yield 2, col, row - 1
    yield 3, col + 1, row
    yield 4, col - 1, row


def compute_action_mask(robot_data: list, obs, config) -> np.ndarray:
    """
    Compute valid action mask for a single robot.

    Args:
        robot_data: [type, col, row, energy, owner, move_cd, jump_cd, build_cd]
        obs: Kaggle observation
        config: Kaggle config

    Returns:
        mask: (13,) bool array where True = action is valid
    """
    rtype, col, row, energy, owner, move_cd, jump_cd, build_cd = robot_data

    mask = np.zeros(13, dtype=bool)

    # IDLE is always valid
    mask[0] = True

    wall = _wall_at(col, row, obs, config)

    # === Movement (actions 1-4) ===
    if move_cd == 0:
        for action_id, adj_col, adj_row in _iter_adjacent(col, row):
            if not (wall & DIRECTION_BITS[action_id]) and _in_bounds(adj_col, adj_row, obs, config):
                mask[action_id] = True

    # === Factory-specific (actions 5-8) ===
    if rtype == 0:  # Factory
        spawn_col, spawn_row = col, row + 1
        spawn_clear = not (wall & 1) and _in_bounds(spawn_col, spawn_row, obs, config)
        if build_cd == 0 and spawn_clear:  # Can build only if spawn cell clear
            if energy >= config.scoutCost:
                mask[5] = True  # BUILD_SCOUT
            if energy >= config.workerCost:
                mask[6] = True  # BUILD_WORKER
            if energy >= config.minerCost:
                mask[7] = True  # BUILD_MINER

        if jump_cd == 0:
            # Check if landing position would be on board
            # JUMP_NORTH by 2 cells
            if _in_bounds(col, row + 2, obs, config):
                mask[8] = True  # JUMP_NORTH

    # === Worker-specific (actions 9-10) ===
    if rtype == 2:  # Worker
        if energy >= config.wallRemoveCost:
            # Check if there's any adjacent wall to remove
            if any(wall & bit for bit in DIRECTION_BITS.values()):
                mask[9] = True  # REMOVE_WALL
            # Check if there's any adjacent open space to build wall
            if any(
                not (wall & DIRECTION_BITS[action_id]) and _in_bounds(adj_col, adj_row, obs, config)
                for action_id, adj_col, adj_row in _iter_adjacent(col, row)
            ):
                mask[10] = True  # BUILD_WALL

    # === Miner-specific (action 11) ===
    if rtype == 3:  # Miner
        mining_key = f"{col},{row}"
        mining_nodes = _get_field(obs, "miningNodes", {})
        transform_cost = _get_field(config, "transformCost", 100)
        if mining_key in mining_nodes and energy >= transform_cost:
            mask[11] = True  # TRANSFORM

    # === TRANSFER (action 12) ===
    # Check if there's adjacent friendly robot
    for _, adj_col, adj_row in _iter_adjacent(col, row):
        for uid, data in _get_field(obs, "robots", {}).items():
            if data[1] == adj_col and data[2] == adj_row and data[4] == owner:
                mask[12] = True
                break

    return mask


def resolve_action_direction_heuristic(
    action_id: int, robot_data: list, obs, config
) -> str:
    """
    Resolve directional actions to specific direction strings.

    For actions like REMOVE_WALL, BUILD_WALL, TRANSFER, JUMP that have implicit
    directions, apply heuristic to pick best direction:
    - REMOVE_WALL: prefer clearing path north (for scrolling pressure)
    - BUILD_WALL: prefer building north (defensive)
    - TRANSFER: transfer to nearest robot with lowest energy
    - JUMP_NORTH: always north

    Args:
        action_id: Unified action ID (0-12)
        robot_data: [type, col, row, energy, owner, move_cd, jump_cd, build_cd]
        obs: Kaggle observation
        config: Kaggle config

    Returns:
        Concrete action string (e.g., "REMOVE_NORTH")
    """
    rtype, col, row, energy, owner, move_cd, jump_cd, build_cd = robot_data

    wall = _wall_at(col, row, obs, config)

    directions = ["NORTH", "EAST", "SOUTH", "WEST"]
    wall_bits = [1, 2, 4, 8]

    if action_id == 9:  # REMOVE_WALL
        # Priority: NORTH > EAST > SOUTH > WEST
        for dir_name, wall_bit in zip(directions, wall_bits):
            if wall & wall_bit:
                return f"REMOVE_{dir_name}"
        return "IDLE"

    elif action_id == 10:  # BUILD_WALL
        # Priority: NORTH > EAST > SOUTH > WEST (build defensive wall)
        for dir_name, wall_bit in zip(directions, wall_bits):
            if not (wall & wall_bit):
                return f"BUILD_{dir_name}"
        return "IDLE"

    elif action_id == 12:  # TRANSFER
        # Find nearest friendly robot with lowest energy
        best_uid = None
        best_dist = float("inf")
        best_energy = float("inf")

        for uid, data in _get_field(obs, "robots", {}).items():
            if data[4] == owner:
                adj_col, adj_row = data[1], data[2]
                if not (adj_col == col and adj_row == row):
                    dist = abs(adj_col - col) + abs(adj_row - row)
                    if dist == 1 and data[3] < best_energy:
                        best_energy = data[3]
                        best_uid = uid

        if best_uid:
            # Determine direction to best_uid
            robots = _get_field(obs, "robots", {})
            adj_col, adj_row = robots[best_uid][1], robots[best_uid][2]
            if adj_row < row:
                return "TRANSFER_NORTH"
            elif adj_row > row:
                return "TRANSFER_SOUTH"
            elif adj_col > col:
                return "TRANSFER_EAST"
            elif adj_col < col:
                return "TRANSFER_WEST"

        return "IDLE"

    elif action_id == 8:  # JUMP_NORTH
        return "JUMP_NORTH"

    return ACTION_ID_TO_STRING.get(action_id, "IDLE")


if __name__ == "__main__":
    print("action_utils.py loaded successfully")
