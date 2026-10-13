"""Shared-pheromone colony environment built on ANTS-RL's maze generator.

This is an experimental extension, separate from the Phase-1 single-ant API.
Each ant receives a local six-channel view (walls, food, nest, food-directed
trail, nest-directed trail, and carrying state). A shared PPO policy chooses
one action per ant; the team reward is food delivered to the nest.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .maze import ACTION_DELTAS, FREE, bfs_distances, generate_maze, place_entities


class ColonyMazeEnv(gym.Env):
    """Cooperative repeated foraging with decentralized local observations.

    Pheromone channel 0 marks paths explored outbound and is followed by
    returning ants; channel 1 marks successful return paths and is followed by
    outbound ants. Diffusion is restricted to free cells and evaporation is
    applied once per synchronous colony step.
    """

    metadata = {"render_modes": []}

    def __init__(self, width=11, height=11, n_ants=4, obs_window=3,
                 episode_steps=None, maze_algorithm="dfs", braid=0.0,
                 pheromones=True, evaporation=0.02, diffusion=0.08,
                 deposit=1.0, max_pheromone=8.0, shaping=True, gamma=0.99,
                 step_penalty=0.01):
        super().__init__()
        self.width, self.height = int(width), int(height)
        self.n_ants, self.obs_window = int(n_ants), int(obs_window)
        self.episode_steps = int(episode_steps or 12 * ((width - 1) * (height - 1) // 2))
        self.maze_algorithm, self.braid = maze_algorithm, float(braid)
        self.pheromones = bool(pheromones)
        self.evaporation, self.diffusion = float(evaporation), float(diffusion)
        self.deposit, self.max_pheromone = float(deposit), float(max_pheromone)
        self.shaping, self.gamma = bool(shaping), float(gamma)
        self.step_penalty = float(step_penalty)
        if self.n_ants < 1 or not 0 <= self.evaporation <= 1 or not 0 <= self.diffusion <= 1:
            raise ValueError("n_ants must be positive; evaporation and diffusion must be in [0, 1]")
        self.size = 2 * self.obs_window + 1
        self.action_space = spaces.MultiDiscrete(np.full(self.n_ants, 4, dtype=np.int64))
        self.observation_space = spaces.Box(0, 1, shape=(self.n_ants, 6, self.size, self.size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.grid = generate_maze(self.width, self.height, self.np_random, self.maze_algorithm, self.braid)
        self.nest, self.food = place_entities(self.grid, self.np_random, min_distance=3)
        self.positions = [self.nest] * self.n_ants
        self.carrying = np.zeros(self.n_ants, dtype=bool)
        self.trails = np.zeros((2, self.width, self.height), dtype=np.float32)
        self.t = 0
        self.deliveries = 0
        self.path_lengths = np.zeros(self.n_ants, dtype=np.int32)
        self.optimal_distance = int(bfs_distances(self.grid, self.food)[self.nest])
        self.dist_food = bfs_distances(self.grid, self.food)
        self.dist_nest = bfs_distances(self.grid, self.nest)
        self.max_distance = max(int(self.dist_food.max()), int(self.dist_nest.max()), 1)
        return self._observe(), self._info()

    def _observe(self):
        k, n = self.obs_window, self.size
        result = np.zeros((self.n_ants, 6, n, n), dtype=np.float32)
        for a, (x, y) in enumerate(self.positions):
            for i, dx in enumerate(range(-k, k + 1)):
                for j, dy in enumerate(range(-k, k + 1)):
                    xx, yy = x + dx, y + dy
                    if not (0 <= xx < self.width and 0 <= yy < self.height):
                        result[a, 0, i, j] = 1.0
                        continue
                    result[a, 0, i, j] = float(self.grid[xx, yy] != FREE)
                    result[a, 3, i, j] = self.trails[0, xx, yy] / self.max_pheromone
                    result[a, 4, i, j] = self.trails[1, xx, yy] / self.max_pheromone
                    result[a, 5, i, j] = float(self.carrying[a])
                    if (xx, yy) == self.food:
                        result[a, 1, i, j] = 1.0
                    if (xx, yy) == self.nest:
                        result[a, 2, i, j] = 1.0
        return result

    def _info(self):
        return {"deliveries": self.deliveries, "steps": self.t,
                "optimal_distance": self.optimal_distance,
                "pheromone_mass": float(self.trails.sum())}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(self.n_ants)
        rewards = -self.step_penalty
        phi_before = self._potential() if self.shaping else 0.0
        for a, action in enumerate(actions):
            x, y = self.positions[a]
            dx, dy = ACTION_DELTAS[int(action)]
            xx, yy = x + dx, y + dy
            if 0 <= xx < self.width and 0 <= yy < self.height and self.grid[xx, yy] == FREE:
                self.positions[a] = (xx, yy)
                self.path_lengths[a] += 1
            if self.pheromones:
                channel = 1 if self.carrying[a] else 0
                self.trails[channel, *self.positions[a]] += self.deposit
            if not self.carrying[a] and self.positions[a] == self.food:
                self.carrying[a] = True
            elif self.carrying[a] and self.positions[a] == self.nest:
                self.carrying[a] = False
                self.deliveries += 1
                rewards += 1.0

        if self.pheromones:
            self._update_trails()
        self.t += 1
        if self.shaping:
            # The step budget is a time-limit truncation, not an absorbing
            # terminal state. Keep the potential and let PPO bootstrap from
            # the final state instead of creating a spurious terminal bonus.
            phi_after = self._potential()
            rewards += self.gamma * phi_after - phi_before
        truncated = self.t >= self.episode_steps
        return self._observe(), rewards, False, truncated, self._info()

    def _potential(self):
        """Team PBRS potential: mean negative geodesic distance to each ant's target."""
        total = 0.0
        for a, pos in enumerate(self.positions):
            distances = self.dist_nest if self.carrying[a] else self.dist_food
            total -= float(distances[pos]) / self.max_distance
        return total / self.n_ants

    def _update_trails(self):
        old = self.trails * (1.0 - self.evaporation)
        new = old * (1.0 - self.diffusion)
        # Four-neighbour, wall-aware diffusion; blocked mass remains in place.
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] != FREE:
                    new[:, x, y] = 0.0
                    continue
                nbrs = [(x + dx, y + dy) for dx, dy in ACTION_DELTAS
                        if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                        and self.grid[x + dx, y + dy] == FREE]
                if nbrs:
                    share = old[:, x, y] * self.diffusion / len(nbrs)
                    for xx, yy in nbrs:
                        new[:, xx, yy] += share
        self.trails = np.clip(new, 0.0, self.max_pheromone)
