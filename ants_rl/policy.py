"""
Feature extractors for the ant observation (SB3 ``BaseFeaturesExtractor``).

Input is ``(C, 2k+1, 2k+1)`` with ``C = 3 * n_stack`` (wall, food, nest per
frame). ``AntCNNExtractor``:

    Conv3x3(C->16) -> ReLU -> Conv3x3(16->32) -> ReLU
    -> AdaptiveAvgPool(3x3) -> Linear(288 -> features_dim) -> ReLU

Adaptive pooling makes the extractor independent of the window size, which is
what allows evaluating a trained policy with the same window on larger mazes.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AntCNNExtractor(BaseFeaturesExtractor):
    """
    Convolutional feature extractor for the 3-channel ant maze observation.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        The env's observation space — shape (3, obs_size, obs_size).
    features_dim : int
        Size of the output embedding fed into the policy/value heads (and LSTM).
        Default 64 works well for an 11×11 maze; increase for larger mazes.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Block 1 — local edge / wall detection
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Block 2 — broader spatial context
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Spatial pooling to a fixed 3×3 regardless of obs_window size
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )

        # Compute flattened CNN output size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_out_dim = self.cnn(sample).shape[1]   # should be 32 * 3 * 3 = 288

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        observations : Tensor  shape (batch, 3, H, W)

        Returns
        -------
        Tensor  shape (batch, features_dim)
        """
        return self.linear(self.cnn(observations))


# ---------------------------------------------------------------------------
# Attention-enhanced extractor  (experimental, Phase 2 placeholder)
# ---------------------------------------------------------------------------

class AntAttentionExtractor(BaseFeaturesExtractor):
    """
    CNN + channel-wise attention extractor.

    Adds a squeeze-and-excitation (SE) block after the first conv layer so the
    network can learn to weight the three semantic channels (walls, food, nest)
    differently based on the agent's current task state.

    Optional drop-in replacement (``model.extractor: attention``).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        se_reduction: int = 4,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Squeeze-and-Excitation attention on the 16-channel feature map
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),            # (16, 1, 1) — global avg per channel
            nn.Flatten(),                       # (16,)
            nn.Linear(16, 16 // se_reduction),  # squeeze
            nn.ReLU(),
            nn.Linear(16 // se_reduction, 16),  # excite
            nn.Sigmoid(),
        )

        # Second conv + pooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            x = self.conv1(sample)
            scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
            x = x * scale
            cnn_out_dim = self.conv2(x).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.conv1(observations)
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)   # channel attention weights
        x = x * scale                                    # rescale feature channels
        x = self.conv2(x)
        return self.linear(x)


EXTRACTORS = {"cnn": AntCNNExtractor, "attention": AntAttentionExtractor}


def make_policy_kwargs(extractor: str = "cnn", features_dim: int = 64,
                       net_arch: list | dict | None = None) -> dict:
    """``policy_kwargs`` for PPO / RecurrentPPO with an ant feature extractor."""
    kw = dict(
        features_extractor_class=EXTRACTORS[extractor],
        features_extractor_kwargs=dict(features_dim=features_dim),
    )
    if net_arch is not None:
        kw["net_arch"] = net_arch
    return kw
