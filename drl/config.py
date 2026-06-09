"""Training configuration for DRL agent."""

from dataclasses import dataclass
from typing import List


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Environment
    opponent: str = "random"  # "random", path to agent module, or comma-separated list
    max_steps: int = 500
    num_envs: int = 8

    # PPO hyperparameters
    learning_rate: float = 3e-4
    learning_rate_decay: str = "cosine"  # "cosine", "linear", or "none"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_steps: int = 2048  # Rollout buffer size
    batch_size: int = 256
    num_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    ent_coef: float = 0.01  # Will be annealed
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Exploration/Entropy
    ent_coef_decay: float = 0.995  # Anneal entropy coefficient

    # Training meta
    total_timesteps: int = 1_000_000
    checkpoint_interval: int = 10_000
    eval_interval: int = 50_000
    eval_episodes: int = 10
    seed: int = 42

    # Model architecture
    cnn_channels: List[int] = None  # Will be [32, 64, 64]
    gru_hidden: int = 256
    fc_hidden: int = 128
    action_dim: int = 13

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_levels: List[dict] = None

    # Self-play
    use_self_play: bool = False
    self_play_pool_size: int = 10
    self_play_snapshot_interval: int = 10_000
    self_play_win_rate_threshold: float = 0.75
    self_play_min_episodes: int = 100

    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    # Logging
    log_interval: int = 10
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    metrics_csv: str = "./logs/training_metrics.csv"
    plot_dir: str = "./logs/plots"
    plot_interval: int = 50_000
    action_histogram_interval: int = 10_000

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 64]

        if self.curriculum_levels is None:
            self.curriculum_levels = [
                {
                    "name": "no_scroll_no_fog",
                    "scroll_speed": 0,
                    "fog": False,
                    "enemy": None,
                    "gate_reward": 5.0,
                },
                {
                    "name": "slow_scroll_no_fog",
                    "scroll_speed": 0.25,
                    "fog": False,
                    "enemy": None,
                    "gate_reward": 5.0,
                },
                {
                    "name": "slow_scroll_fog",
                    "scroll_speed": 0.25,
                    "fog": True,
                    "enemy": None,
                    "gate_reward": 5.0,
                },
                {
                    "name": "slow_scroll_fog_random",
                    "scroll_speed": 0.25,
                    "fog": True,
                    "enemy": "random",
                    "gate_reward": 3.0,
                },
                {
                    "name": "full_speed",
                    "scroll_speed": 1.0,
                    "fog": True,
                    "enemy": "random",
                    "gate_reward": 0.0,  # No gate, train to end
                },
            ]


@dataclass
class RewardConfig:
    """Reward shaping coefficients."""

    win: float = 10.0
    loss: float = -10.0
    draw: float = 0.0

    crystal: float = 1.0  # Multiplied by (value / 50)
    mine_income: float = 0.1  # Per mine per turn
    mine_built: float = 3.0
    exploration: float = 0.02  # Per new cell

    crush_win: float = 1.5
    robot_loss: float = -0.5
    friendly_fire: float = -3.0

    survival: float = 0.01  # Per turn factory alive
    scroll_danger: float = -0.1  # Per robot near southBound
    scroll_death: float = -5.0

    energy_delta: float = 0.001  # 0.001 * delta_energy per turn
    idle_penalty: float = -0.05  # Per robot idle >5 turns


# Default configs
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_REWARD_CONFIG = RewardConfig()


if __name__ == "__main__":
    config = TrainConfig()
    print(f"Default config: {config}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Checkpoint interval: {config.checkpoint_interval:,}")
