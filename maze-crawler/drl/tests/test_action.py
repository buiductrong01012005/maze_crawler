"""Tests for action masking and resolution utilities."""

import pytest
import numpy as np
from drl.action_utils import (
    compute_action_mask,
    ACTION_ID_TO_STRING,
    STRING_TO_ACTION_ID,
)


class MockConfig:
    width = 20
    scoutCost = 50
    workerCost = 200
    minerCost = 300
    wallRemoveCost = 100
    transformCost = 100


class MockObs:
    def __init__(self):
        self.walls = np.full((20 * 20,), 0, dtype=int)  # No walls by default
        self.robots = {}
        self.miningNodes = {}
        self.southBound = 0
        self.northBound = 20


def test_action_id_to_string():
    """Test action ID to string mapping."""
    assert ACTION_ID_TO_STRING[0] == "IDLE"
    assert ACTION_ID_TO_STRING[1] == "NORTH"
    assert ACTION_ID_TO_STRING[5] == "BUILD_SCOUT"
    assert ACTION_ID_TO_STRING[11] == "TRANSFORM"
    print("✓ test_action_id_to_string passed")


def test_string_to_action_id():
    """Test string to action ID mapping."""
    assert STRING_TO_ACTION_ID["IDLE"] == 0
    assert STRING_TO_ACTION_ID["NORTH"] == 1
    assert STRING_TO_ACTION_ID["BUILD_SCOUT"] == 5
    print("✓ test_string_to_action_id passed")


def test_compute_action_mask_idle_always_valid():
    """Test that IDLE is always valid."""
    config = MockConfig()
    obs = MockObs()

    robot_data = [1, 10, 10, 100, 0, 1, 0, 0]  # Scout with move_cd=1
    mask = compute_action_mask(robot_data, obs, config)

    # IDLE (action 0) should always be valid
    assert mask[0] == True
    print("✓ test_compute_action_mask_idle_always_valid passed")


def test_compute_action_mask_movement_blocked_by_cooldown():
    """Test movement blocked by cooldown."""
    config = MockConfig()
    obs = MockObs()

    robot_data = [1, 10, 10, 100, 0, 1, 0, 0]  # Scout with move_cd=1
    mask = compute_action_mask(robot_data, obs, config)

    # NORTH/SOUTH/EAST/WEST should be False due to cooldown
    assert mask[1] == False  # NORTH
    assert mask[2] == False  # SOUTH
    assert mask[3] == False  # EAST
    assert mask[4] == False  # WEST
    print("✓ test_compute_action_mask_movement_blocked_by_cooldown passed")


def test_compute_action_mask_movement_available():
    """Test movement available when cooldown is 0."""
    config = MockConfig()
    obs = MockObs()

    robot_data = [1, 10, 10, 100, 0, 0, 0, 0]  # Scout with move_cd=0
    mask = compute_action_mask(robot_data, obs, config)

    # Movement should be available
    assert mask[1] == True  # NORTH
    assert mask[2] == True  # SOUTH
    assert mask[3] == True  # EAST
    assert mask[4] == True  # WEST
    print("✓ test_compute_action_mask_movement_available passed")


def test_compute_action_mask_factory_build():
    """Test Factory build actions."""
    config = MockConfig()
    obs = MockObs()

    # Factory with enough energy and no cooldown
    robot_data = [0, 10, 10, 1000, 0, 0, 0, 0]
    mask = compute_action_mask(robot_data, obs, config)

    # BUILD_SCOUT (action 5) should be valid
    assert mask[5] == True  # BUILD_SCOUT
    assert mask[6] == True  # BUILD_WORKER
    assert mask[7] == True  # BUILD_MINER
    print("✓ test_compute_action_mask_factory_build passed")


def test_compute_action_mask_factory_insufficient_energy():
    """Test Factory cannot build with insufficient energy."""
    config = MockConfig()
    obs = MockObs()

    # Factory with insufficient energy
    robot_data = [0, 10, 10, 10, 0, 0, 0, 0]  # Only 10 energy
    mask = compute_action_mask(robot_data, obs, config)

    # BUILD actions should be False
    assert mask[5] == False  # BUILD_SCOUT (costs 50)
    assert mask[6] == False  # BUILD_WORKER (costs 200)
    print("✓ test_compute_action_mask_factory_insufficient_energy passed")


def test_compute_action_mask_factory_jump():
    """Test Factory JUMP action."""
    config = MockConfig()
    obs = MockObs()

    # Factory with jump_cd=0
    robot_data = [0, 10, 10, 1000, 0, 0, 0, 0]
    mask = compute_action_mask(robot_data, obs, config)

    # JUMP_NORTH (action 8) should be valid
    assert mask[8] == True
    print("✓ test_compute_action_mask_factory_jump passed")


def test_compute_action_mask_worker_wall_actions():
    """Test Worker wall actions."""
    config = MockConfig()
    obs = MockObs()

    # Worker with enough energy
    robot_data = [2, 10, 10, 300, 0, 0, 0, 0]
    mask = compute_action_mask(robot_data, obs, config)

    # REMOVE_WALL and BUILD_WALL should be available
    # (depends on wall configuration, but with default 0 walls both should be valid)
    assert mask[9] == False  # REMOVE_WALL (no walls to remove)
    assert mask[10] == True  # BUILD_WALL (open space to build)
    print("✓ test_compute_action_mask_worker_wall_actions passed")


def test_compute_action_mask_transfer():
    """Test TRANSFER action."""
    config = MockConfig()
    obs = MockObs()

    # Scout at (10, 10)
    robot_data = [1, 10, 10, 100, 0, 0, 0, 0]

    # Add another friendly robot adjacent to robot
    obs.robots["robot2"] = [2, 10, 11, 200, 0, 0, 0, 0]  # Worker at (10, 11)

    mask = compute_action_mask(robot_data, obs, config)

    # TRANSFER (action 12) should be valid
    assert mask[12] == True
    print("✓ test_compute_action_mask_transfer passed")


def test_compute_action_mask_scout_cannot_build():
    """Test Scout cannot perform Factory-only actions."""
    config = MockConfig()
    obs = MockObs()

    # Scout
    robot_data = [1, 10, 10, 100, 0, 0, 0, 0]
    mask = compute_action_mask(robot_data, obs, config)

    # BUILD actions should be False for Scout
    assert mask[5] == False  # BUILD_SCOUT
    assert mask[6] == False  # BUILD_WORKER
    assert mask[7] == False  # BUILD_MINER
    assert mask[8] == False  # JUMP_NORTH

    # Movement should be valid
    assert mask[1] == True  # NORTH
    print("✓ test_compute_action_mask_scout_cannot_build passed")


if __name__ == "__main__":
    test_action_id_to_string()
    test_string_to_action_id()
    test_compute_action_mask_idle_always_valid()
    test_compute_action_mask_movement_blocked_by_cooldown()
    test_compute_action_mask_movement_available()
    test_compute_action_mask_factory_build()
    test_compute_action_mask_factory_insufficient_energy()
    test_compute_action_mask_factory_jump()
    test_compute_action_mask_worker_wall_actions()
    test_compute_action_mask_transfer()
    test_compute_action_mask_scout_cannot_build()
    print("\n✅ All action tests passed!")
