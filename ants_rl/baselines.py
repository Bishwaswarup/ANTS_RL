"""
Non-learned reference policies. Each exposes ``reset(env)`` and ``act(env)``.

* ``RandomPolicy``       - uniform random action (lower bound).
* ``WallFollowerPolicy`` - right-hand rule using only the 4 adjacent cells
  (local information that is inside the agent's window) plus a heading,
  i.e. a single integer of memory. In a perfect maze it is guaranteed to
  reach every cell, so it always succeeds given enough steps.
* ``OraclePolicy``       - follows the true shortest path (privileged; SPL=1).
"""

from __future__ import annotations

import numpy as np

from .maze import ACTION_DELTAS, FREE


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env) -> int:
        return int(self.rng.integers(4))


class WallFollowerPolicy:
    name = "wall_follower"

    def reset(self, env):
        self.heading = 0

    @staticmethod
    def _free(env, a):
        x, y = env.agent_pos
        dx, dy = ACTION_DELTAS[a]
        nx, ny = x + dx, y + dy
        return 0 <= nx < env.width and 0 <= ny < env.height and env.grid[nx, ny] == FREE

    def act(self, env) -> int:
        # Actions are ordered clockwise (up, right, down, left): right turn = +1.
        for turn in (1, 0, -1, 2):
            a = (self.heading + turn) % 4
            if self._free(env, a):
                self.heading = a
                return a
        return self.heading


class OraclePolicy:
    name = "oracle"

    def reset(self, env):
        pass

    def act(self, env) -> int:
        x, y = env.agent_pos
        d = env.dist_to_food[x, y]
        for a, (dx, dy) in enumerate(ACTION_DELTAS):
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.width and 0 <= ny < env.height and env.dist_to_food[nx, ny] == d - 1:
                return a
        return 0


BASELINES = {"random": RandomPolicy, "wall_follower": WallFollowerPolicy, "oracle": OraclePolicy}
