"""Tests for environment wrapper."""

import pytest
from drl.env_wrapper import CrawlGymnasiumEnv


def test_env_reset():
    """Test environment reset."""
    env = CrawlGymnasiumEnv(opponent="random", debug=False)
    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert info is not None
    assert "step" in info
    print("✓ test_env_reset passed")


def test_env_step():
    """Test environment step."""
    env = CrawlGymnasiumEnv(opponent="random", debug=False)
    obs, _ = env.reset()

    if obs:  # If there are robots
        actions = {uid: 0 for uid in obs}  # IDLE action
        obs_new, reward, terminated, truncated, info = env.step(actions)

        assert isinstance(obs_new, dict) or obs_new == {}
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "step" in info
    print("✓ test_env_step passed")


def test_env_episode():
    """Test full episode."""
    env = CrawlGymnasiumEnv(opponent="random", debug=False)
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0

    for step in range(100):
        if not obs:
            break

        # Random action for each robot
        actions = {uid: 0 for uid in obs}  # IDLE
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    assert steps > 0
    print(f"✓ test_env_episode passed (steps={steps}, reward={total_reward:.3f})")


def test_env_smoke_test():
    """Smoke test: run 10 episodes without crashing."""
    env = CrawlGymnasiumEnv(opponent="random", debug=False)

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0

        for _ in range(500):
            if not obs:
                break

            actions = {uid: 0 for uid in obs}
            obs, reward, terminated, truncated, _ = env.step(actions)
            steps += 1

            if terminated or truncated:
                done = True
                break

        assert done or steps > 0

    print("✓ test_env_smoke_test passed (10 episodes)")


if __name__ == "__main__":
    test_env_reset()
    test_env_step()
    test_env_episode()
    test_env_smoke_test()
    print("\n✅ All environment tests passed!")
