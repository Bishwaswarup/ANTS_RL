"""
analysis/metrics.py
===================
Episode metrics collection and summary statistics for EAC experiments.

Typical usage
-------------
::

    from analysis.metrics import MetricsCollector

    collector = MetricsCollector()

    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep = collector.start_episode()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep.record_step(
                reward=reward,
                pos=tuple(env.agent_pos),
                hit_wall=info.get("hit_wall", False),
            )
            done = terminated or truncated

        collector.end_episode(success=terminated)

    summary = collector.summary()
    print(summary)
    collector.save("analysis/results/eval_metrics.json")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-episode recorder
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecorder:
    """Live recorder for a single episode.  Returned by MetricsCollector.start_episode()."""

    episode_id: int
    start_time: float = field(default_factory=time.time)

    # Accumulated per-step data
    _rewards: list[float] = field(default_factory=list)
    _positions: list[tuple] = field(default_factory=list)
    _wall_hits: int = 0

    def record_step(
        self,
        reward: float,
        pos: tuple,
        hit_wall: bool = False,
    ) -> None:
        """Record one environment step."""
        self._rewards.append(reward)
        self._positions.append(pos)
        if hit_wall:
            self._wall_hits += 1

    def finalise(self, success: bool, optimal_steps: Optional[int] = None) -> "EpisodeMetrics":
        """Convert to a frozen EpisodeMetrics object."""
        n_steps = len(self._rewards)
        unique_cells = len(set(self._positions))
        efficiency = (
            optimal_steps / max(n_steps, 1)
            if (optimal_steps is not None and success)
            else 0.0
        )
        return EpisodeMetrics(
            episode_id=self.episode_id,
            n_steps=n_steps,
            total_reward=float(np.sum(self._rewards)),
            success=success,
            path_efficiency=min(efficiency, 1.0),
            unique_cells=unique_cells,
            wall_hits=self._wall_hits,
            wall_hit_rate=self._wall_hits / max(n_steps, 1),
            duration_s=time.time() - self.start_time,
            positions=self._positions,
        )


# ---------------------------------------------------------------------------
# Frozen per-episode metrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    """Immutable summary of a completed episode."""

    episode_id: int
    n_steps: int
    total_reward: float
    success: bool
    path_efficiency: float   # optimal_steps / actual_steps  (0 if failed)
    unique_cells: int
    wall_hits: int
    wall_hit_rate: float     # wall_hits / n_steps
    duration_s: float
    positions: list[tuple]   # Full trajectory for heatmap generation

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = [list(p) for p in d["positions"]]
        return d


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Collects and aggregates metrics across multiple evaluation episodes.

    Parameters
    ----------
    run_name : str  Label for this evaluation run (used in saved files).
    """

    def __init__(self, run_name: str = "eval") -> None:
        self.run_name = run_name
        self._episodes: list[EpisodeMetrics] = []
        self._current: Optional[EpisodeRecorder] = None
        self._episode_counter: int = 0

    # ---- Episode lifecycle ----

    def start_episode(self) -> EpisodeRecorder:
        """Begin recording a new episode.  Returns the active recorder."""
        self._episode_counter += 1
        self._current = EpisodeRecorder(episode_id=self._episode_counter)
        return self._current

    def end_episode(
        self,
        success: bool,
        optimal_steps: Optional[int] = None,
    ) -> EpisodeMetrics:
        """
        Finalise the current episode and store its metrics.

        Parameters
        ----------
        success       : bool  Whether the agent reached its target.
        optimal_steps : int   BFS shortest path for efficiency calculation.
        """
        if self._current is None:
            raise RuntimeError("Call start_episode() before end_episode().")
        metrics = self._current.finalise(success=success, optimal_steps=optimal_steps)
        self._episodes.append(metrics)
        self._current = None
        return metrics

    # ---- Aggregated statistics ----

    @property
    def n(self) -> int:
        return len(self._episodes)

    @property
    def success_rate(self) -> float:
        if not self._episodes:
            return 0.0
        return sum(e.success for e in self._episodes) / self.n

    @property
    def mean_steps(self) -> float:
        return float(np.mean([e.n_steps for e in self._episodes])) if self._episodes else 0.0

    @property
    def mean_reward(self) -> float:
        return float(np.mean([e.total_reward for e in self._episodes])) if self._episodes else 0.0

    @property
    def mean_efficiency(self) -> float:
        successes = [e for e in self._episodes if e.success]
        if not successes:
            return 0.0
        return float(np.mean([e.path_efficiency for e in successes]))

    @property
    def mean_unique_cells(self) -> float:
        return float(np.mean([e.unique_cells for e in self._episodes])) if self._episodes else 0.0

    @property
    def mean_wall_hit_rate(self) -> float:
        return float(np.mean([e.wall_hit_rate for e in self._episodes])) if self._episodes else 0.0

    def visit_heatmap(self, width: int, height: int) -> np.ndarray:
        """
        Build a visitation heatmap from all recorded trajectories.

        Returns
        -------
        np.ndarray  Shape (width, height) — raw visit counts.
        """
        heatmap = np.zeros((width, height), dtype=np.int32)
        for ep in self._episodes:
            for x, y in ep.positions:
                if 0 <= x < width and 0 <= y < height:
                    heatmap[x, y] += 1
        return heatmap

    def summary(self) -> dict:
        """Return a flat dict of all aggregate statistics."""
        return {
            "run_name": self.run_name,
            "n_episodes": self.n,
            "success_rate": round(self.success_rate, 4),
            "mean_steps": round(self.mean_steps, 2),
            "mean_reward": round(self.mean_reward, 4),
            "mean_path_efficiency": round(self.mean_efficiency, 4),
            "mean_unique_cells": round(self.mean_unique_cells, 2),
            "mean_wall_hit_rate": round(self.mean_wall_hit_rate, 4),
        }

    def save(self, path: str | Path) -> None:
        """
        Persist all episode metrics to a JSON file.

        Parameters
        ----------
        path : str | Path  Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self._episodes],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Metrics saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MetricsCollector":
        """Load a previously saved metrics file."""
        with open(path) as f:
            data = json.load(f)

        collector = cls(run_name=data["summary"].get("run_name", "loaded"))
        for ep_data in data["episodes"]:
            ep_data["positions"] = [tuple(p) for p in ep_data["positions"]]
            collector._episodes.append(EpisodeMetrics(**ep_data))

        return collector

    def __repr__(self) -> str:
        return (
            f"MetricsCollector('{self.run_name}', "
            f"n={self.n}, "
            f"success={self.success_rate:.1%})"
        )
