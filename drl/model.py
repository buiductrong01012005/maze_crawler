"""CNN-GRU Actor-Critic model for Crawl DRL agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrawlActorCritic(nn.Module):
    """
    Shared CNN-GRU policy network for decentralized multi-agent control.

    Architecture:
    - CNN: 3 layers (12→32→64→64) for spatial feature extraction
    - GRU: 256 hidden units for handling partial observability
    - Actor head: outputs logits for 13 actions (with masking)
    - Critic head: outputs state value estimate
    """

    def __init__(
        self,
        input_channels: int = 12,
        map_size: int = 11,
        scalar_dim: int = 13,
        cnn_channels: list = None,
        gru_hidden: int = 256,
        fc_hidden: int = 128,
        action_dim: int = 13,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device
        self.gru_hidden = gru_hidden
        self.action_dim = action_dim

        if cnn_channels is None:
            cnn_channels = [32, 64, 64]

        # === CNN for spatial feature extraction ===
        # Input: (B, 12, 11, 11)
        self.conv1 = nn.Conv2d(input_channels, cnn_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])

        self.conv2 = nn.Conv2d(
            cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(cnn_channels[1])

        self.conv3 = nn.Conv2d(
            cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(cnn_channels[2])

        # Flatten CNN output: cnn_channels[-1] * map_size * map_size
        cnn_flat_dim = cnn_channels[-1] * map_size * map_size

        # FC layer after CNN + scalars concat
        combined_dim = cnn_flat_dim + scalar_dim
        self.fc_shared = nn.Linear(combined_dim, 256)

        # === GRU for temporal/memory processing ===
        self.gru = nn.GRUCell(256, gru_hidden)

        # === Actor Head ===
        self.actor_fc1 = nn.Linear(gru_hidden, fc_hidden)
        self.actor_fc2 = nn.Linear(fc_hidden, action_dim)

        # === Critic Head ===
        self.critic_fc1 = nn.Linear(gru_hidden, fc_hidden)
        self.critic_fc2 = nn.Linear(fc_hidden, 1)

        self.to(device)

    def forward(
        self,
        local_map: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for actor-critic network.

        Args:
            local_map: (B, 12, 11, 11) local view channels
            scalars: (B, 13) scalar features
            action_mask: (B, 13) binary mask for valid actions
            hidden_state: (B, gru_hidden) previous GRU hidden state

        Returns:
            action_probs: (B, 13) action probabilities
            value: (B, 1) state value estimate
            new_hidden: (B, gru_hidden) new GRU hidden state
        """
        B = local_map.size(0)

        # === CNN feature extraction ===
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(local_map)))

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten: (B, cnn_channels[-1] * 11 * 11)
        x = x.view(B, -1)

        # === Concatenate with scalars ===
        x = torch.cat([x, scalars], dim=-1)

        # === Shared FC ===
        x = F.relu(self.fc_shared(x))

        # === GRU ===
        if hidden_state is None:
            hidden_state = torch.zeros(
                B, self.gru_hidden, dtype=x.dtype, device=x.device
            )

        new_hidden = self.gru(x, hidden_state)

        # === Actor Head ===
        actor_x = F.relu(self.actor_fc1(new_hidden))
        logits = self.actor_fc2(actor_x)

        # Apply action masking: set invalid actions to -inf
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))

        # Softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)

        # Handle NaN from -inf (all masked)
        action_probs = torch.where(
            torch.isnan(action_probs),
            torch.ones_like(action_probs) / self.action_dim,
            action_probs,
        )

        # === Critic Head ===
        critic_x = F.relu(self.critic_fc1(new_hidden))
        value = self.critic_fc2(critic_x)

        return action_probs, value, new_hidden

    def get_action_and_value(
        self,
        local_map: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action samples and their log probabilities, plus value.

        Args:
            local_map: (B, 12, 11, 11)
            scalars: (B, 13)
            action_mask: (B, 13) binary mask
            hidden_state: Previous GRU state

        Returns:
            action: (B,) sampled action indices
            log_prob: (B,) log probability of sampled action
            value: (B, 1) state value
            new_hidden: (B, gru_hidden) new hidden state
        """
        action_probs, value, new_hidden = self.forward(
            local_map, scalars, action_mask, hidden_state
        )

        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1), new_hidden

    def state_dict_for_export(self):
        """Return state dict without GRU hidden state."""
        return super().state_dict()


if __name__ == "__main__":
    # Test model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrawlActorCritic(device=device)

    # Test forward pass
    B = 4  # batch size
    local_map = torch.randn(B, 12, 11, 11, device=device)
    scalars = torch.randn(B, 13, device=device)
    action_mask = torch.ones(B, 13, dtype=torch.bool, device=device)

    action_probs, value, hidden = model(local_map, scalars, action_mask)

    print(f"Action probs shape: {action_probs.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("Model test passed!")
