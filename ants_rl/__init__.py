"""ANTS-RL: partially observable maze foraging with PPO."""

from .env import AntMazeEnv, REWARD_PRESETS  # noqa: F401
from .colony import ColonyMazeEnv  # noqa: F401

__version__ = "0.2.0"
