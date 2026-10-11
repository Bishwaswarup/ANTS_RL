"""
AntMazeEnv: single-agent, partially observable maze foraging.

An ant starts at the nest and must reach the food. It observes only a
``(2k+1) x (2k+1)`` egocentric window with three binary channels
(wall, food, nest). Cells outside the grid are reported as walls.

Layout distribution
-------------------
``n_train_mazes`` controls how many distinct layouts the agent can see:

* ``n_train_mazes = N > 0``  -> a fixed pool of N layouts (the first N seeds
  >= ``maze_seed_base`` whose layout is in the "train" split), sampled
  uniformly each episode.
* ``n_train_mazes = 0``      -> a fresh layout every episode (procedural),
  seeds drawn from ``[PROCEDURAL_SEED_BASE, 2**31)``, rejecting "test" layouts.

Nest/food positions are re-sampled every episode in both cases. Layouts are
split into train/test by a hash of the layout itself (``maze.layout_split``),
so held-out evaluation layouts are guaranteed never to occur in training.

An episode can be pinned exactly with
``env.reset(options={"maze_seed": s, "placement_seed": p})``.

Rewards
-------
See ``REWARD_PRESETS``. ``"pbrs"`` uses the potential
``Phi(s) = -w * d(s) / D`` where ``d`` is the geodesic (BFS) distance to the
food and ``D`` the largest such distance in the current maze, and adds
``F = gamma * Phi(s') - Phi(s)`` (Ng et al., 1999). ``Phi = 0`` at the goal,
so the shaping is policy-invariant for the episodic task. ``"legacy"``
reproduces the original repository reward (Manhattan-distance difference
without ``gamma``, wall and revisit penalties), which is *not*
policy-invariant.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .maze import (ACTION_DELTAS, FREE, bfs_distances, generate_maze, layout_split,
                   place_entities, seeds_in_split)

TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 1_000_000
PROCEDURAL_SEED_BASE = 10_000_000

REWARD_PRESETS: dict[str, dict] = {
    # +1 on reaching food, nothing else.
    "sparse": dict(success=1.0, step_penalty=0.0, wall_penalty=0.0,
                   shaping="none", shaping_weight=0.0,
                   revisit_penalty=0.0, revisit_cap=0),
    # Sparse reward + exact potential-based shaping on geodesic distance.
    "pbrs": dict(success=1.0, step_penalty=0.0, wall_penalty=0.0,
                 shaping="pbrs_geodesic", shaping_weight=1.0,
                 revisit_penalty=0.0, revisit_cap=0),
    # Original ANTS reward (v0): not policy-invariant.
    "legacy": dict(success=10.0, step_penalty=0.01, wall_penalty=0.10,
                   shaping="manhattan_diff", shaping_weight=0.15,
                   revisit_penalty=0.05, revisit_cap=5),
}


def resolve_reward(reward_cfg: dict | str | None) -> dict:
    """Merge a reward config (``{"preset": name, **overrides}``) onto its preset."""
    if reward_cfg is None:
        reward_cfg = {"preset": "pbrs"}
    if isinstance(reward_cfg, str):
        reward_cfg = {"preset": reward_cfg}
    cfg = dict(reward_cfg)
    preset = cfg.pop("preset", "pbrs")
    if preset not in REWARD_PRESETS:
        raise ValueError(f"Unknown reward preset {preset!r}")
    out = copy.deepcopy(REWARD_PRESETS[preset])
    unknown = set(cfg) - set(out)
    if unknown:
        raise ValueError(f"Unknown reward keys: {sorted(unknown)}")
    out.update(cfg)
    out["preset"] = preset
    return out


class AntMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    def __init__(
        self,
        width: int = 11,
        height: int = 11,
        obs_window: int = 3,
        min_food_dist: int = 3,
        max_steps: int | None = None,
        n_train_mazes: int = 0,
        maze_seed_base: int = TRAIN_SEED_BASE,
        maze_algorithm: str = "dfs",
        reward: dict | str | None = "pbrs",
        gamma: float = 0.99,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.width, self.height = width, height
        self.obs_window = obs_window
        self.min_food_dist = min_food_dist
        self.n_train_mazes = n_train_mazes
        self.maze_seed_base = maze_seed_base
        self.maze_algorithm = maze_algorithm
        self.reward_cfg = resolve_reward(reward)
        self.gamma = gamma
        self.render_mode = render_mode

        n_rooms = ((width - 1) // 2) * ((height - 1) // 2)
        self.n_free = 2 * n_rooms - 1  # perfect maze: rooms + (rooms - 1) passages
        self.max_steps = max_steps if max_steps is not None else 6 * self.n_free

        self.obs_size = 2 * obs_window + 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(3, self.obs_size, self.obs_size), dtype=np.float32
        )
        self._maze_cache: dict[int, np.ndarray] = {}
        self._pool = (seeds_in_split(n_train_mazes, maze_seed_base, "train", width, height,
                                     maze_algorithm) if n_train_mazes > 0 else None)

    # ------------------------------------------------------------------ utils
    def _maze(self, seed: int) -> np.ndarray:
        if seed not in self._maze_cache:
            if len(self._maze_cache) > 4096:
                self._maze_cache.clear()
            rng = np.random.default_rng(seed)
            self._maze_cache[seed] = generate_maze(
                self.width, self.height, rng, self.maze_algorithm
            )
        return self._maze_cache[seed]

    def _sample_maze_seed(self) -> int:
        if self._pool is not None:
            return self._pool[int(self.np_random.integers(len(self._pool)))]
        while True:
            seed = int(self.np_random.integers(PROCEDURAL_SEED_BASE, 2**31 - 1))
            if layout_split(self._maze(seed)) == "train":
                return seed

    def _potential(self, pos) -> float:
        rc = self.reward_cfg
        if rc["shaping"] == "pbrs_geodesic":
            return -rc["shaping_weight"] * self.dist_to_food[pos] / self.max_dist
        return 0.0

    # -------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.maze_seed = int(options.get("maze_seed", self._sample_maze_seed()))
        self.grid = self._maze(self.maze_seed)

        if "placement_seed" in options:
            prng = np.random.default_rng(int(options["placement_seed"]))
        else:
            prng = self.np_random
        self.nest_pos, self.food_pos = place_entities(self.grid, prng, self.min_food_dist)

        self.dist_to_food = bfs_distances(self.grid, self.food_pos)
        self.max_dist = max(int(self.dist_to_food.max()), 1)
        self.optimal_steps = int(self.dist_to_food[self.nest_pos])

        self.agent_pos = self.nest_pos
        self.step_count = 0
        self.wall_hits = 0
        self.visit_counts = {self.agent_pos: 1}
        return self._get_obs(), self._info()

    def _get_obs(self) -> np.ndarray:
        k = self.obs_window
        ax, ay = self.agent_pos
        # Pad by k with walls so the window can be sliced directly.
        padded = np.pad(self.grid, k, constant_values=1)
        obs = np.zeros((3, self.obs_size, self.obs_size), dtype=np.float32)
        obs[0] = padded[ax : ax + 2 * k + 1, ay : ay + 2 * k + 1]
        for ch, (ex, ey) in ((1, self.food_pos), (2, self.nest_pos)):
            i, j = ex - ax + k, ey - ay + k
            if 0 <= i < self.obs_size and 0 <= j < self.obs_size:
                obs[ch, i, j] = 1.0
        return obs

    def _info(self, success: bool = False) -> dict:
        return {
            "maze_seed": self.maze_seed,
            "optimal_steps": self.optimal_steps,
            "steps": self.step_count,
            "wall_hits": self.wall_hits,
            "unique_cells": len(self.visit_counts),
            "is_success": success,
        }

    def step(self, action):
        rc = self.reward_cfg
        self.step_count += 1
        old = self.agent_pos
        dx, dy = ACTION_DELTAS[int(action)]
        nx, ny = old[0] + dx, old[1] + dy
        hit_wall = not (0 <= nx < self.width and 0 <= ny < self.height) or self.grid[nx, ny] != FREE
        new = old if hit_wall else (nx, ny)
        self.agent_pos = new
        self.wall_hits += int(hit_wall)
        self.visit_counts[new] = self.visit_counts.get(new, 0) + 1

        success = new == self.food_pos
        reward = -rc["step_penalty"]
        if hit_wall:
            reward -= rc["wall_penalty"]
        if rc["shaping"] == "pbrs_geodesic":
            phi_next = 0.0 if success else self._potential(new)
            reward += self.gamma * phi_next - self._potential(old)
        elif rc["shaping"] == "manhattan_diff":
            d_old = abs(old[0] - self.food_pos[0]) + abs(old[1] - self.food_pos[1])
            d_new = abs(new[0] - self.food_pos[0]) + abs(new[1] - self.food_pos[1])
            reward += rc["shaping_weight"] * (d_old - d_new)
        if rc["revisit_penalty"] > 0:
            extra = min(self.visit_counts[new] - 1, rc["revisit_cap"])
            reward -= rc["revisit_penalty"] * max(extra, 0)
        if success:
            reward += rc["success"]

        terminated = success
        truncated = (not terminated) and self.step_count >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, self._info(success)

    # ------------------------------------------------------------- rendering
    def render(self, cell: int = 24, trail: set | None = None) -> np.ndarray:
        """Return an RGB image (H, W, 3) of the full maze state."""
        colors = {
            "wall": (40, 40, 40), "free": (245, 245, 245), "trail": (255, 205, 205),
            "nest": (139, 69, 19), "food": (50, 170, 50), "ant": (210, 30, 60),
            "view": (215, 230, 250),
        }
        img = np.zeros((self.height * cell, self.width * cell, 3), dtype=np.uint8)
        ax, ay = self.agent_pos
        k = self.obs_window
        for x in range(self.width):
            for y in range(self.height):
                c = colors["wall"] if self.grid[x, y] else colors["free"]
                if not self.grid[x, y] and abs(x - ax) <= k and abs(y - ay) <= k:
                    c = colors["view"]
                if trail and (x, y) in trail and not self.grid[x, y]:
                    c = colors["trail"]
                if (x, y) == self.nest_pos:
                    c = colors["nest"]
                if (x, y) == self.food_pos:
                    c = colors["food"]
                img[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = c
        cy, cx = ay * cell + cell // 2, ax * cell + cell // 2
        yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= (cell // 3) ** 2] = colors["ant"]
        return img


def make_env_fn(env_cfg: dict, gamma: float = 0.99, seed: int | None = None):
    """Factory returning a thunk, for SB3 vec envs."""
    def _thunk():
        env = AntMazeEnv(gamma=gamma, **env_cfg)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk
