"""
agents/memory.py
================
Working memory and episodic memory modules for EAC agents.

Overview
--------
Biological ant colonies rely on two kinds of memory:

  1. **Working memory** — short-term state held within a single foraging trip
     (e.g. "I just came from the east").  In the RL setting this is handled
     implicitly by the LSTM's hidden state, but we also expose an explicit
     spatial working memory that can be concatenated to the observation.

  2. **Episodic memory** — cross-episode accumulation of experience.  Not used
     during the LSTM's own forward pass, but useful for curriculum control
     (e.g. increase maze difficulty when recent success rate exceeds 80 %).

Both modules are *optional* add-ons — the Phase 1 baseline does not use them.
They are designed to slot into Phase 2 multi-agent experiments.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional


# ---------------------------------------------------------------------------
# Spatial Working Memory
# ---------------------------------------------------------------------------

class SpatialMemory:
    """
    A 2-D binary map of cells visited in the current episode.

    The agent receives this as an additional observation channel, giving it
    an explicit "I've been here before" signal that complements the LSTM.

    Shape of the memory map: (width, height) — same as the maze grid.

    Usage
    -----
    ::

        mem = SpatialMemory(width=11, height=11, decay=0.95)
        mem.reset()

        # After each env step:
        mem.update(agent_pos=(3, 5))

        # Get as an extra observation channel (7×7 patch around agent):
        patch = mem.get_patch(cx=3, cy=5, window=3)   # shape (7, 7)
    """

    def __init__(
        self,
        width: int,
        height: int,
        decay: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        width, height : int    Maze dimensions.
        decay         : float  Multiplicative decay per step (1.0 = no decay,
                               0.95 = recent visits shine brighter).
        """
        self.width = width
        self.height = height
        self.decay = decay
        self._map = np.zeros((width, height), dtype=np.float32)

    def reset(self) -> None:
        """Clear memory at episode start."""
        self._map[:] = 0.0

    def update(self, agent_pos: tuple) -> None:
        """
        Mark the current position as visited and decay older marks.

        Parameters
        ----------
        agent_pos : (x, y)
        """
        if self.decay < 1.0:
            self._map *= self.decay
        x, y = agent_pos
        self._map[x, y] = 1.0

    def get_patch(self, cx: int, cy: int, window: int = 3) -> np.ndarray:
        """
        Return a (2*window+1) × (2*window+1) patch of the memory map.

        Out-of-bounds cells are 0 (unvisited / outside maze).
        """
        size = 2 * window + 1
        patch = np.zeros((size, size), dtype=np.float32)

        for i, dx in enumerate(range(-window, window + 1)):
            for j, dy in enumerate(range(-window, window + 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    patch[i, j] = self._map[nx, ny]

        return patch

    @property
    def coverage(self) -> float:
        """Fraction of maze cells visited this episode (0–1)."""
        total_cells = (self._map > 0).sum()
        return float(total_cells) / (self.width * self.height)

    def __repr__(self) -> str:
        return f"SpatialMemory({self.width}×{self.height}, decay={self.decay})"


# ---------------------------------------------------------------------------
# Episodic Memory  (cross-episode curriculum control)
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Circular buffer of episode outcomes used for curriculum decisions.

    Tracks recent success rate and mean episode length so a training loop can
    automatically adjust task difficulty (e.g. increase min_food_dist when the
    agent is consistently succeeding).

    Example
    -------
    ::

        mem = EpisodicMemory(window=50)

        # At each episode end:
        mem.record(success=True, length=87, total_reward=8.3)

        if mem.should_increase_difficulty(threshold=0.75):
            env.min_food_dist = min(env.min_food_dist + 1, 10)
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._successes: deque[bool] = deque(maxlen=window)
        self._lengths: deque[int] = deque(maxlen=window)
        self._rewards: deque[float] = deque(maxlen=window)

    def record(
        self,
        success: bool,
        length: int,
        total_reward: float,
    ) -> None:
        """Record the outcome of a completed episode."""
        self._successes.append(success)
        self._lengths.append(length)
        self._rewards.append(total_reward)

    @property
    def n(self) -> int:
        return len(self._successes)

    @property
    def success_rate(self) -> float:
        if not self._successes:
            return 0.0
        return sum(self._successes) / len(self._successes)

    @property
    def mean_length(self) -> float:
        if not self._lengths:
            return 0.0
        return float(np.mean(list(self._lengths)))

    @property
    def mean_reward(self) -> float:
        if not self._rewards:
            return 0.0
        return float(np.mean(list(self._rewards)))

    def should_increase_difficulty(self, threshold: float = 0.75) -> bool:
        """
        Return True when the recent success rate exceeds `threshold`
        AND the buffer is full (we have enough data to decide).
        """
        return self.n == self.window and self.success_rate >= threshold

    def should_decrease_difficulty(self, threshold: float = 0.25) -> bool:
        """
        Return True when the recent success rate falls below `threshold`.
        Prevents the curriculum from advancing too fast.
        """
        return self.n == self.window and self.success_rate < threshold

    def summary(self) -> dict:
        return {
            "memory/success_rate": self.success_rate,
            "memory/mean_length": self.mean_length,
            "memory/mean_reward": self.mean_reward,
            "memory/n_episodes": self.n,
        }

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(window={self.window}, "
            f"n={self.n}, "
            f"success_rate={self.success_rate:.1%})"
        )
