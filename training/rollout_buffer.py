"""
training/rollout_buffer.py
==========================
Custom rollout buffer that wraps Stable-Baselines3's DictRolloutBuffer and
adds per-episode statistics useful for tracking ANTS training progress.

Why extend the buffer?
----------------------
SB3's built-in buffer stores raw transitions efficiently for PPO updates but
discards episode-level information (success rate, episode length, path
efficiency).  This subclass intercepts episode boundaries and accumulates
statistics that feed into TensorBoard and the analysis module.

Usage
-----
Pass `buffer_class=TrackedRolloutBuffer` when constructing the model in a
custom training loop.  The extra stats are available via `buffer.episode_stats`
after each call to `collect_rollouts`.

Note
----
For standard Phase 1 training (train.py), SB3's default buffer is used.
This class is designed for custom training loops in Phase 2 / research runs.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Episode statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Per-episode statistics captured during rollout collection."""

    episode_id: int
    length: int                 # Total steps taken
    total_reward: float         # Undiscounted cumulative reward
    success: bool               # Whether the agent reached its target
    path_efficiency: float      # Optimal steps / actual steps  (0–1; 1 = perfect)
    unique_cells_visited: int   # Count of distinct cells visited
    wall_hits: int              # Times the agent tried to walk into a wall

    @property
    def is_efficient(self) -> bool:
        """True if the agent solved the maze in ≤2× the optimal path length."""
        return self.success and self.path_efficiency >= 0.5


@dataclass
class RolloutStats:
    """Aggregated statistics over the most recent window of episodes."""

    window: int = 100
    _episodes: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(self, ep: EpisodeStats) -> None:
        self._episodes.append(ep)

    # ---- Aggregation helpers ----

    @property
    def n(self) -> int:
        return len(self._episodes)

    @property
    def success_rate(self) -> float:
        if not self._episodes:
            return 0.0
        return sum(e.success for e in self._episodes) / len(self._episodes)

    @property
    def mean_episode_length(self) -> float:
        if not self._episodes:
            return 0.0
        return np.mean([e.length for e in self._episodes])

    @property
    def mean_reward(self) -> float:
        if not self._episodes:
            return 0.0
        return np.mean([e.total_reward for e in self._episodes])

    @property
    def mean_path_efficiency(self) -> float:
        successes = [e for e in self._episodes if e.success]
        if not successes:
            return 0.0
        return np.mean([e.path_efficiency for e in successes])

    @property
    def mean_unique_cells(self) -> float:
        if not self._episodes:
            return 0.0
        return np.mean([e.unique_cells_visited for e in self._episodes])

    def summary(self) -> dict[str, float]:
        """Return a flat dict suitable for logging to TensorBoard."""
        return {
            "rollout/success_rate": self.success_rate,
            "rollout/mean_ep_length": self.mean_episode_length,
            "rollout/mean_reward": self.mean_reward,
            "rollout/mean_path_efficiency": self.mean_path_efficiency,
            "rollout/mean_unique_cells": self.mean_unique_cells,
            "rollout/n_episodes": self.n,
        }

    def __repr__(self) -> str:
        return (
            f"RolloutStats(n={self.n}, "
            f"success={self.success_rate:.1%}, "
            f"mean_len={self.mean_episode_length:.1f}, "
            f"efficiency={self.mean_path_efficiency:.2f})"
        )


# ---------------------------------------------------------------------------
# Episode tracker (used inside a training loop)
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """
    Lightweight per-episode bookkeeper.

    Attach one of these to an environment wrapper or custom training loop
    to automatically build EpisodeStats at episode boundaries.

    Example
    -------
    ::

        tracker = EpisodeTracker()
        stats_log = RolloutStats()

        obs, _ = env.reset()
        tracker.on_reset(agent_pos=env.agent_pos)

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.on_step(
                reward=reward,
                new_pos=env.agent_pos,
                hit_wall=info.get("hit_wall", False),
            )
            if terminated or truncated:
                ep_stats = tracker.on_episode_end(success=terminated)
                stats_log.record(ep_stats)
                obs, _ = env.reset()
                tracker.on_reset(agent_pos=env.agent_pos)
    """

    _episode_counter: int = 0

    def __init__(self, optimal_steps: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        optimal_steps : int, optional
            Known shortest path length for the current maze (from BFS).
            Used to compute path_efficiency.  If None, efficiency = NaN.
        """
        self.optimal_steps = optimal_steps
        self._reset_state()

    def on_reset(self, agent_pos: tuple) -> None:
        """Call at the start of every episode (after env.reset())."""
        self._reset_state()
        self._visited.add(tuple(agent_pos))

    def on_step(
        self,
        reward: float,
        new_pos: tuple,
        hit_wall: bool = False,
    ) -> None:
        """Call after every env.step()."""
        self._steps += 1
        self._total_reward += reward
        self._visited.add(tuple(new_pos))
        if hit_wall:
            self._wall_hits += 1

    def on_episode_end(self, success: bool) -> EpisodeStats:
        """
        Call when an episode terminates or truncates.

        Returns
        -------
        EpisodeStats  — record it in a RolloutStats instance.
        """
        EpisodeTracker._episode_counter += 1
        efficiency = (
            self.optimal_steps / max(self._steps, 1)
            if (self.optimal_steps is not None and success)
            else 0.0
        )
        return EpisodeStats(
            episode_id=EpisodeTracker._episode_counter,
            length=self._steps,
            total_reward=self._total_reward,
            success=success,
            path_efficiency=min(efficiency, 1.0),
            unique_cells_visited=len(self._visited),
            wall_hits=self._wall_hits,
        )

    # ---- Internal ----

    def _reset_state(self) -> None:
        self._steps: int = 0
        self._total_reward: float = 0.0
        self._visited: set = set()
        self._wall_hits: int = 0
