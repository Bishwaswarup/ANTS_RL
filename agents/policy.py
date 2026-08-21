"""
agents/policy.py
================
Custom feature extractor for the EAC ant observation space.

Observation format (from AntMazeEnv)
--------------------------------------
Shape: (3, obs_size, obs_size)  where obs_size = 2 * obs_window + 1 = 7

  Channel 0: Wall mask           — 1 where there is a wall, else 0
  Channel 1: Food indicator      — 1 at the food cell (if in window), else 0
  Channel 2: Nest indicator      — 1 at the nest cell (if in window), else 0

Why a custom extractor?
-----------------------
SB3's default MlpExtractor treats the observation as a flat vector, losing
all spatial structure.  A small CNN preserves spatial relationships (e.g.
"wall to the left, food up-right") which is critical for navigating mazes.

The extractor is compatible with SB3's RecurrentPPO via the `policy_kwargs`
argument::

    from agents.policy import AntCNNExtractor

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=AntCNNExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        ),
    )

Architecture
------------
  Input  (3, 7, 7)
  → Conv2d(3→16, kernel=3, padding=1) + ReLU    # (16, 7, 7)
  → Conv2d(16→32, kernel=3, padding=1) + ReLU   # (32, 7, 7)
  → AdaptiveAvgPool2d(3, 3)                      # (32, 3, 3) — spatial pyramid
  → Flatten                                       # (288,)
  → Linear(288 → features_dim) + ReLU            # (features_dim,)

The LSTM in RecurrentPPO sits on top of this extractor, giving the agent
memory across timesteps — crucial for navigating partially-observable mazes.
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

        n_channels = observation_space.shape[0]   # 3

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

    This is experimental and not yet used in Phase 1 training.
    It is provided here as a drop-in replacement for ablation studies.
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


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_policy_kwargs(
    extractor: str = "cnn",
    features_dim: int = 64,
) -> dict:
    """
    Build the `policy_kwargs` dict for RecurrentPPO.

    Parameters
    ----------
    extractor    : "cnn" | "attention"
    features_dim : int

    Example
    -------
    ::

        model = RecurrentPPO(
            "MlpLstmPolicy", env,
            **make_policy_kwargs(extractor="cnn", features_dim=64),
        )
    """
    extractor_cls = {
        "cnn": AntCNNExtractor,
        "attention": AntAttentionExtractor,
    }[extractor]

    return dict(
        policy_kwargs=dict(
            features_extractor_class=extractor_cls,
            features_extractor_kwargs=dict(features_dim=features_dim),
        )
    )
