"""Self-play league manager for training."""

import os
import shutil
from typing import List, Optional
from pathlib import Path


class SelfPlayLeague:
    """
    Manages a pool of opponent models for self-play training.

    Maintains a rolling pool of older model snapshots to train against,
    preventing overfitting and promoting emergent behaviors.
    """

    def __init__(
        self,
        pool_size: int = 10,
        snapshot_interval: int = 100_000,
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Args:
            pool_size: Maximum number of models to keep in pool
            snapshot_interval: Frequency of snapshots (in steps)
            checkpoint_dir: Directory to save checkpoints
        """
        self.pool_size = pool_size
        self.snapshot_interval = snapshot_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pool: List[str] = []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_opponent(self) -> Optional[str]:
        """
        Select opponent for next episode.

        Returns:
            Path to opponent model, or "random" for random agent
        """
        import random

        if not self.pool or random.random() < 0.2:
            # 20% chance to fight random
            return "random"

        return random.choice(self.pool)

    def maybe_snapshot(self, checkpoint_path: str, step: int) -> bool:
        """
        Possibly add current model to pool.

        Args:
            checkpoint_path: Path to model checkpoint
            step: Current training step

        Returns:
            True if snapshot was added, False otherwise
        """
        if step % self.snapshot_interval != 0:
            return False

        # Copy checkpoint to pool directory
        pool_name = f"opponent_step_{step}.pt"
        pool_path = str(self.checkpoint_dir / pool_name)

        try:
            shutil.copy(checkpoint_path, pool_path)
            self.add_checkpoint(pool_path)
            return True
        except Exception as e:
            print(f"Failed to snapshot: {e}")
            return False

    def add_checkpoint(self, checkpoint_path: str, remove_old: bool = True):
        """Add an existing checkpoint file to the opponent pool."""
        self.pool.append(str(checkpoint_path))

        while len(self.pool) > self.pool_size:
            old_path = self.pool.pop(0)
            if remove_old and os.path.exists(old_path):
                os.remove(old_path)

    def list_opponents(self) -> List[str]:
        """List all available opponents in pool."""
        return self.pool.copy()


if __name__ == "__main__":
    league = SelfPlayLeague(pool_size=5)
    print("Self-play league initialized")
