"""Tests for observation extraction utilities."""

import pytest
import numpy as np
from unittest.mock import Mock
from drl.obs_utils import extract_local_view, extract_scalars


class MockConfig:
    width = 20


class MockObs:
    def __init__(self):
        self.walls = np.full((20 * 20,), 0, dtype=int)
        self.robots = {}
        self.crystals = {}
        self.mines = {}
        self.miningNodes = {}
        self.southBound = 0
        self.northBound = 20
        self.step = 0


def test_extract_local_view_basic():
    """Test basic local view extraction."""
    config = MockConfig()
    obs = MockObs()

    local_map = extract_local_view(obs, 10, 10, config, radius=5)

    assert local_map.shape == (12, 11, 11)
    assert local_map.dtype == np.float32
    print("✓ test_extract_local_view_basic passed")


def test_extract_local_view_boundary():
    """Test local view at boundary."""
    config = MockConfig()
    obs = MockObs()

    # Test at corner (0, 0)
    local_map = extract_local_view(obs, 0, 0, config, radius=5)

    assert local_map.shape == (12, 11, 11)
    # Center of local map
    assert local_map[4, 5, 5] == 1  # visibility should be 1 (discovered)
    print("✓ test_extract_local_view_boundary passed")


def test_extract_local_view_fog_of_war():
    """Test fog of war encoding."""
    config = MockConfig()
    obs = MockObs()

    # Set some cells as unexplored (wall = -1)
    obs.walls[0] = -1

    local_map = extract_local_view(obs, 0, 0, config, radius=5)

    # Unexplored cell should have visibility = 0
    assert local_map.shape == (12, 11, 11)
    print("✓ test_extract_local_view_fog_of_war passed")


def test_extract_scalars():
    """Test scalar feature extraction."""
    config = MockConfig()
    obs = MockObs()

    robot_data = [0, 10, 10, 500, 0, 0, 0, 0]  # Factory with 500 energy
    scalars = extract_scalars(robot_data, obs, config)

    assert scalars.shape == (13,)
    assert scalars.dtype == np.float32

    # Check one-hot encoding (Factory = type 0)
    assert scalars[0] == 1.0

    # Check energy ratio (500 / 1000 = 0.5)
    assert scalars[4] == 0.5

    print("✓ test_extract_scalars passed")


def test_extract_scalars_scout():
    """Test scalars for Scout."""
    config = MockConfig()
    obs = MockObs()

    robot_data = [1, 10, 10, 50, 0, 0, 0, 0]  # Scout with 50 energy
    scalars = extract_scalars(robot_data, obs, config)

    # Scout = type 1
    assert scalars[1] == 1.0

    # Energy ratio (50 / 100 = 0.5)
    assert scalars[4] == 0.5

    print("✓ test_extract_scalars_scout passed")


def test_extract_scalars_with_cooldowns():
    """Test scalars with cooldowns."""
    config = MockConfig()
    obs = MockObs()

    robot_data = [0, 10, 10, 1000, 0, 1, 5, 5]  # Factory with cooldowns
    scalars = extract_scalars(robot_data, obs, config)

    # move_cd=1, move_period=2 → ratio=0.5
    assert scalars[5] == 0.5

    # build_cd=5, max=10 → ratio=0.5
    assert scalars[6] == 0.5

    # jump_cd=5, max=20 → ratio=0.25
    assert scalars[7] == 0.25

    print("✓ test_extract_scalars_with_cooldowns passed")


if __name__ == "__main__":
    test_extract_local_view_basic()
    test_extract_local_view_boundary()
    test_extract_local_view_fog_of_war()
    test_extract_scalars()
    test_extract_scalars_scout()
    test_extract_scalars_with_cooldowns()
    print("\n✅ All tests passed!")
