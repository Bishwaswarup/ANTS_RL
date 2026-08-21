"""
envs/pheromone.py
=================
Pheromone diffusion grid for Emergent Ant Colony (EAC) Phase 2.

Two independent pheromone channels are supported out of the box:
  - Channel 0: "food trail"   — deposited while carrying food, guides others to food
  - Channel 1: "return trail" — deposited while returning to nest, guides others home

Physics
-------
Each timestep the grid evolves as:

    P[t+1] = (P[t] * (1 - evaporation_rate) convolved with diffusion_kernel).clip(0, max_strength)

Deposits are additive: P[x, y] += deposit_amount at the agent's current cell.

Usage (Phase 2 multi-agent env)
---------------------------------
    grid = PheromoneGrid(width=11, height=11, n_channels=2)

    # Agent deposits a food-trail pheromone at position (3, 5)
    grid.deposit(x=3, y=5, channel=0, amount=1.0)

    # Update physics every environment step
    grid.step()

    # Read pheromone strength at a position
    strength = grid.read(x=3, y=5, channel=0)

    # Get a normalised observation patch (for adding to agent observation space)
    patch = grid.get_patch(cx=3, cy=5, window=3, channel=0)
"""

import numpy as np
from scipy.ndimage import convolve


# Isotropic 3×3 diffusion kernel (centre = 1 − sum_neighbours * rate)
_DIFFUSION_KERNEL = np.array([
    [0.05, 0.10, 0.05],
    [0.10, 0.40, 0.10],
    [0.05, 0.10, 0.05],
], dtype=np.float32)
# Rows sum to 1, so no mass is created during diffusion.


class PheromoneGrid:
    """
    A multi-channel pheromone grid with evaporation and diffusion.

    Parameters
    ----------
    width, height : int
        Maze dimensions (must match the environment grid).
    n_channels : int
        Number of independent pheromone types (default 2).
    evaporation_rate : float
        Fraction of pheromone that evaporates each step (0–1).
    diffusion_rate : float
        Fraction that spreads to neighbours each step (0–1).
    deposit_amount : float
        Default deposit strength per agent action.
    max_strength : float
        Ceiling clamp to prevent unbounded accumulation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        n_channels: int = 2,
        evaporation_rate: float = 0.05,
        diffusion_rate: float = 0.10,
        deposit_amount: float = 1.0,
        max_strength: float = 10.0,
    ) -> None:
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.evaporation_rate = evaporation_rate
        self.diffusion_rate = diffusion_rate
        self.deposit_amount = deposit_amount
        self.max_strength = max_strength

        # Shape: (n_channels, width, height)  — float32 for speed
        self._grid = np.zeros((n_channels, width, height), dtype=np.float32)

        # Pre-build the diffusion kernel scaled by diffusion_rate
        self._kernel = self._build_kernel(diffusion_rate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all pheromone trails (call at episode start)."""
        self._grid[:] = 0.0

    def deposit(self, x: int, y: int, channel: int = 0, amount: float | None = None) -> None:
        """
        Add pheromone at grid cell (x, y) on the given channel.

        Parameters
        ----------
        x, y    : int   Cell coordinates (column, row).
        channel : int   Pheromone channel index.
        amount  : float Deposit strength; defaults to self.deposit_amount.
        """
        if amount is None:
            amount = self.deposit_amount
        self._grid[channel, x, y] = min(
            self._grid[channel, x, y] + amount, self.max_strength
        )

    def step(self) -> None:
        """
        Advance pheromone physics by one timestep.

        Order of operations (matches standard ACO literature):
          1. Evaporation  — uniform decay across the grid
          2. Diffusion    — local spreading to neighbours
          3. Clamp        — enforce [0, max_strength]
        """
        for c in range(self.n_channels):
            # 1. Evaporate
            self._grid[c] *= (1.0 - self.evaporation_rate)

            # 2. Diffuse (scipy convolve wraps edges with 'reflect' mode)
            self._grid[c] = convolve(
                self._grid[c], self._kernel, mode="reflect"
            )

            # 3. Clamp
            np.clip(self._grid[c], 0.0, self.max_strength, out=self._grid[c])

    def read(self, x: int, y: int, channel: int = 0) -> float:
        """Return the raw pheromone strength at (x, y) for a channel."""
        return float(self._grid[channel, x, y])

    def read_normalised(self, x: int, y: int, channel: int = 0) -> float:
        """Return pheromone strength normalised to [0, 1]."""
        return float(self._grid[channel, x, y]) / self.max_strength

    def get_patch(
        self, cx: int, cy: int, window: int = 3, channel: int = 0
    ) -> np.ndarray:
        """
        Return a (2*window+1) × (2*window+1) normalised patch centred on (cx, cy).

        Out-of-bounds cells are filled with 0 (no pheromone outside the maze).
        This patch can be concatenated to the agent's CNN observation.
        """
        size = 2 * window + 1
        patch = np.zeros((size, size), dtype=np.float32)

        for i, dx in enumerate(range(-window, window + 1)):
            for j, dy in enumerate(range(-window, window + 1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    patch[i, j] = self._grid[channel, nx, ny] / self.max_strength

        return patch

    def get_observation(self, cx: int, cy: int, window: int = 3) -> np.ndarray:
        """
        Return stacked patches for ALL channels — shape (n_channels, 2w+1, 2w+1).

        Suitable for appending to the environment's observation tensor.
        """
        patches = np.stack(
            [self.get_patch(cx, cy, window, c) for c in range(self.n_channels)],
            axis=0,
        )
        return patches  # (n_channels, size, size)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Raw pheromone grid — shape (n_channels, width, height)."""
        return self._grid

    def channel_max(self, channel: int = 0) -> float:
        """Peak pheromone strength on a given channel (useful for logging)."""
        return float(self._grid[channel].max())

    def channel_mean(self, channel: int = 0) -> float:
        """Mean pheromone strength on a given channel."""
        return float(self._grid[channel].mean())

    def total_pheromone(self) -> float:
        """Sum across all channels — a global measure of trail activity."""
        return float(self._grid.sum())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_kernel(diffusion_rate: float) -> np.ndarray:
        """
        Build a 3×3 convolution kernel that:
          - Preserves (1 - diffusion_rate) of the centre cell's value.
          - Distributes diffusion_rate evenly across the 4 cardinal neighbours.
          - Distributes half as much to the 4 diagonal neighbours.

        The kernel sums to 1 so total pheromone mass is conserved during diffusion
        (only evaporation removes mass).
        """
        diag = diffusion_rate * 0.5 / 4   # 4 diagonal neighbours
        card = diffusion_rate * 0.5 / 4   # 4 cardinal neighbours get equal share
        # Total outflow = 8 * (diffusion_rate / 8) = diffusion_rate  ✓
        centre = 1.0 - diffusion_rate

        kernel = np.array([
            [diag, card, diag],
            [card, centre, card],
            [diag, card, diag],
        ], dtype=np.float32)

        return kernel


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def make_pheromone_grid(cfg: dict) -> PheromoneGrid:
    """
    Build a PheromoneGrid from the project's YAML config dict.

    Example::

        import yaml
        cfg = yaml.safe_load(open("configs/default.yaml"))
        ph = make_pheromone_grid(cfg["pheromone"])
    """
    ph_cfg = cfg.get("pheromone", {})
    env_cfg = cfg.get("env", {})
    return PheromoneGrid(
        width=env_cfg.get("width", 11),
        height=env_cfg.get("height", 11),
        n_channels=cfg.get("mappo", {}).get("pheromone_channels", 2),
        evaporation_rate=ph_cfg.get("evaporation_rate", 0.05),
        diffusion_rate=ph_cfg.get("diffusion_rate", 0.10),
        deposit_amount=ph_cfg.get("deposit_amount", 1.0),
        max_strength=ph_cfg.get("max_strength", 10.0),
    )
