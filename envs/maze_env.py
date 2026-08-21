import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .maze_generator import MazeGenerator


class AntMazeEnv(gym.Env):
    """
    Phase 1: Single-agent foraging environment.

    Reward structure
    ----------------
    - Step penalty          : -0.01 per step (time pressure)
    - Wall collision        : -0.10 (discourage bumping)
    - Distance shaping      : ±0.15 × Δ(Manhattan dist to food)
    - Revisitation penalty  : -0.05 per extra visit, CAPPED at 5 visits
                              (prevents the unbounded penalty explosion that
                               causes the agent to freeze rather than explore)
    - Food reward           : +10.0 on success (terminal)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Maximum extra-visit penalty per step (5 visits × 0.05 = 0.25 max)
    REVISIT_PENALTY   = 0.05
    REVISIT_CAP       = 5      # stop counting after this many extra visits
    WALL_PENALTY      = 0.10
    STEP_PENALTY      = 0.01
    DISTANCE_WEIGHT   = 0.15
    FOOD_REWARD       = 10.0

    def __init__(self, width=11, height=11, obs_window=3, min_food_dist=3):
        super().__init__()
        self.width = width
        self.height = height
        self.obs_window = obs_window
        self.min_food_dist = min_food_dist

        self.action_space = spaces.Discrete(4)

        self.obs_size = (self.obs_window * 2 + 1)
        # Channels: 0=Wall, 1=Food, 2=Nest
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.obs_size, self.obs_size),
            dtype=np.float32
        )

        self.generator = MazeGenerator(self.width, self.height)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.generator.generate()
        self.nest_pos, self.food_pos = self.generator.place_entities(
            self.grid, min_distance=self.min_food_dist
        )

        self.agent_pos = list(self.nest_pos)
        self.has_food = False
        self.step_count = 0
        self.max_steps = 300

        # Track cell visits in this episode to penalize oscillation loops
        self.visited_counts = {tuple(self.agent_pos): 1}

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((3, self.obs_size, self.obs_size), dtype=np.float32)
        ax, ay = self.agent_pos
        w = self.obs_window

        for i, dx in enumerate(range(-w, w + 1)):
            for j, dy in enumerate(range(-w, w + 1)):
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    obs[0, i, j] = self.grid[nx, ny]
                    if (nx, ny) == self.food_pos and not self.has_food:
                        obs[1, i, j] = 1.0
                    if (nx, ny) == self.nest_pos:
                        obs[2, i, j] = 1.0
                else:
                    obs[0, i, j] = 1.0
        return obs

    def step(self, action):
        self.step_count += 1
        reward = -0.01  # Existential step penalty
        terminated = False
        truncated = False

        old_pos = list(self.agent_pos)
        old_dist = abs(old_pos[0] - self.food_pos[0]) + abs(old_pos[1] - self.food_pos[1])

        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dx = 1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1

        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        # Wall / boundary collision check
        hit_wall = (
            not (0 <= nx < self.width and 0 <= ny < self.height)
            or self.grid[nx, ny] == 1
        )
        if hit_wall:
            reward -= self.WALL_PENALTY
        else:
            self.agent_pos = [nx, ny]

        new_dist = abs(self.agent_pos[0] - self.food_pos[0]) + abs(self.agent_pos[1] - self.food_pos[1])

        # 1. Distance shaping: reward moving closer, penalise moving away
        reward += (old_dist - new_dist) * self.DISTANCE_WEIGHT

        # 2. Revisitation penalty — CAPPED to prevent reward explosion.
        #    Old bug: penalty grew as 0.03 × visit_count, causing −300 rewards
        #    at step 147 and teaching the agent to freeze.
        #    Fix: flat penalty per extra visit, capped at REVISIT_CAP visits.
        pos_tuple = tuple(self.agent_pos)
        self.visited_counts[pos_tuple] = self.visited_counts.get(pos_tuple, 0) + 1
        extra_visits = min(self.visited_counts[pos_tuple] - 1, self.REVISIT_CAP)
        if extra_visits > 0:
            reward -= self.REVISIT_PENALTY * extra_visits

        # Target reached
        if tuple(self.agent_pos) == self.food_pos and not self.has_food:
            reward += self.FOOD_REWARD
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        info = {"hit_wall": hit_wall, "step": self.step_count}
        return self._get_obs(), reward, terminated, truncated, info