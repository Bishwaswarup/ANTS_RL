"""
training/reward_shaping.py
==========================
Composable reward-shaping components for the EAC project.

Design
------
Every shaper is a callable that receives the environment's transition tuple
and returns a scalar delta to ADD to the base reward.  Composing them is
just summing their deltas:

    shapers = [DistanceShaper(weight=0.15), RevisitationPenalty(penalty=0.03)]
    extra   = sum(s(transition) for s in shapers)
    reward += extra

This keeps the reward logic out of the environment class and makes it easy to
ablate individual components (simply drop them from the list).

Transition namedtuple
---------------------
All shapers accept a single `Transition` object so their signatures stay
uniform and new fields can be added without breaking existing shapers.

    Transition(
        old_pos   : tuple[int, int],
        new_pos   : tuple[int, int],
        action    : int,
        has_food  : bool,
        food_pos  : tuple[int, int],
        nest_pos  : tuple[int, int],
        visited   : dict[tuple, int],   # cell → visit count this episode
        step      : int,
        hit_wall  : bool,
    )
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import Protocol


# ---------------------------------------------------------------------------
# Transition data class
# ---------------------------------------------------------------------------

Transition = namedtuple(
    "Transition",
    [
        "old_pos",   # (x, y) before the step
        "new_pos",   # (x, y) after the step  (== old_pos if wall hit)
        "action",    # discrete action taken
        "has_food",  # whether the agent is currently carrying food
        "food_pos",  # current food location
        "nest_pos",  # nest location (fixed per episode)
        "visited",   # dict mapping cell tuples to visit counts
        "step",      # current step number
        "hit_wall",  # True if the action tried to move into a wall
    ],
)


# ---------------------------------------------------------------------------
# Base protocol / interface
# ---------------------------------------------------------------------------

class RewardShaper(Protocol):
    """Any callable that maps a Transition → float delta reward."""

    def __call__(self, t: Transition) -> float:
        ...


# ---------------------------------------------------------------------------
# Concrete shapers
# ---------------------------------------------------------------------------

class StepPenalty:
    """
    Constant per-step cost — the simplest time-pressure signal.

    Encourages the agent to solve the task quickly rather than meandering.

    Parameters
    ----------
    penalty : float  (should be negative, default -0.01)
    """

    def __init__(self, penalty: float = -0.01) -> None:
        self.penalty = penalty

    def __call__(self, t: Transition) -> float:
        return self.penalty

    def __repr__(self) -> str:
        return f"StepPenalty(penalty={self.penalty})"


class WallPenalty:
    """
    Extra penalty applied when the agent attempts to walk into a wall.

    Works *in addition* to the movement being blocked; it discourages
    repeated wall-bumping without completely preventing exploration.
    """

    def __init__(self, penalty: float = -0.10) -> None:
        self.penalty = penalty

    def __call__(self, t: Transition) -> float:
        return self.penalty if t.hit_wall else 0.0

    def __repr__(self) -> str:
        return f"WallPenalty(penalty={self.penalty})"


class DistanceShaper:
    """
    Potential-based distance shaping: reward moving closer to the target,
    penalise moving away.

    Uses Manhattan distance as the potential function.  By the potential-
    based shaping theorem (Ng et al. 1999), this preserves the optimal
    policy of the original MDP while accelerating learning.

    Target switches automatically:
      - No food → target is food_pos
      - Carrying food → target is nest_pos  (for Phase 2 return task)
    """

    def __init__(self, weight: float = 0.15) -> None:
        self.weight = weight

    def _target(self, t: Transition) -> tuple[int, int]:
        return t.nest_pos if t.has_food else t.food_pos

    def __call__(self, t: Transition) -> float:
        target = self._target(t)
        old_dist = abs(t.old_pos[0] - target[0]) + abs(t.old_pos[1] - target[1])
        new_dist = abs(t.new_pos[0] - target[0]) + abs(t.new_pos[1] - target[1])
        return (old_dist - new_dist) * self.weight

    def __repr__(self) -> str:
        return f"DistanceShaper(weight={self.weight})"


class RevisitationPenalty:
    """
    Penalises revisiting already-explored cells to break oscillation loops.

    The penalty grows linearly with the number of extra visits so that
    frequently-revisited cells quickly become very costly.

    Delta = -penalty * max(0, visit_count - 1)
    """

    def __init__(self, penalty: float = 0.03) -> None:
        self.penalty = penalty

    def __call__(self, t: Transition) -> float:
        count = t.visited.get(t.new_pos, 0)
        extra_visits = max(0, count - 1)
        return -self.penalty * extra_visits

    def __repr__(self) -> str:
        return f"RevisitationPenalty(penalty={self.penalty})"


class ExplorationBonus:
    """
    Intrinsic motivation: reward visiting a cell for the first time.

    Inspired by count-based exploration (Bellemare et al. 2016).
    Bonus = bonus_amount / sqrt(visit_count) — quickly decays for
    frequently-visited cells, keeping it focused on genuine novelty.

    Parameters
    ----------
    bonus_amount : float   Scaling factor (default 0.05).
    min_visits   : int     Only award the bonus until this many visits.
    """

    def __init__(self, bonus_amount: float = 0.05, min_visits: int = 1) -> None:
        self.bonus_amount = bonus_amount
        self.min_visits = min_visits

    def __call__(self, t: Transition) -> float:
        count = t.visited.get(t.new_pos, 0)
        if count <= self.min_visits:
            return self.bonus_amount / math.sqrt(max(count, 1))
        return 0.0

    def __repr__(self) -> str:
        return f"ExplorationBonus(bonus={self.bonus_amount}, min_visits={self.min_visits})"


class PheromoneFollowingBonus:
    """
    (Phase 2) Reward the agent for moving to cells with stronger pheromone.

    Incentivises emergent trail-following without hard-coding the behaviour.

    Parameters
    ----------
    pheromone_grid : PheromoneGrid  The shared pheromone layer.
    channel        : int            Which channel to follow (0=food, 1=return).
    weight         : float          Scaling factor for the bonus.
    """

    def __init__(self, pheromone_grid, channel: int = 0, weight: float = 0.05) -> None:
        self.pheromone_grid = pheromone_grid
        self.channel = channel
        self.weight = weight

    def __call__(self, t: Transition) -> float:
        old_strength = self.pheromone_grid.read_normalised(*t.old_pos, self.channel)
        new_strength = self.pheromone_grid.read_normalised(*t.new_pos, self.channel)
        return (new_strength - old_strength) * self.weight

    def __repr__(self) -> str:
        return f"PheromoneFollowingBonus(channel={self.channel}, weight={self.weight})"


# ---------------------------------------------------------------------------
# Composed shaper factory
# ---------------------------------------------------------------------------

class ComposedShaper:
    """
    Runs a list of shapers and returns the sum of their deltas.

    Example
    -------
    ::

        shaper = ComposedShaper([
            StepPenalty(-0.01),
            WallPenalty(-0.10),
            DistanceShaper(0.15),
            RevisitationPenalty(0.03),
        ])
        reward += shaper(transition)
    """

    def __init__(self, shapers: list[RewardShaper]) -> None:
        self.shapers = shapers

    def __call__(self, t: Transition) -> float:
        return sum(s(t) for s in self.shapers)

    def describe(self) -> str:
        lines = ["ComposedShaper(["]
        for s in self.shapers:
            lines.append(f"    {s!r},")
        lines.append("])")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ComposedShaper(n={len(self.shapers)})"


def make_phase1_shaper(cfg: dict | None = None) -> ComposedShaper:
    """
    Build the standard Phase 1 shaper from config values.

    Parameters
    ----------
    cfg : dict  The 'reward' section of default.yaml (or None for defaults).
    """
    r = cfg or {}
    return ComposedShaper([
        StepPenalty(r.get("step_penalty", -0.01)),
        WallPenalty(r.get("wall_penalty", -0.10)),
        DistanceShaper(r.get("distance_weight", 0.15)),
        RevisitationPenalty(r.get("revisit_penalty", 0.03)),
    ])
