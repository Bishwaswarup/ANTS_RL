"""
Maze generation and grid utilities.

Conventions
-----------
* Grids are ``np.int8`` arrays indexed ``grid[x, y]`` with shape ``(width, height)``.
* ``1`` = wall, ``0`` = free cell.
* Dimensions must be odd; "rooms" live on odd coordinates and the outer
  border is always wall.

All randomness goes through an explicit ``numpy.random.Generator`` so that a
maze is a pure function of its seed.
"""

from __future__ import annotations

import hashlib
from collections import deque
from functools import lru_cache

import numpy as np

WALL = 1
FREE = 0

# Action convention shared by the env, baselines and tests:
# 0 = up (y-1), 1 = right (x+1), 2 = down (y+1), 3 = left (x-1)
ACTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _check_dims(width: int, height: int) -> None:
    if width < 5 or height < 5 or width % 2 == 0 or height % 2 == 0:
        raise ValueError(f"Maze dims must be odd and >= 5, got {width}x{height}")


def generate_maze(
    width: int,
    height: int,
    rng: np.random.Generator,
    algorithm: str = "dfs",
    braid: float = 0.0,
) -> np.ndarray:
    """Generate a maze. With ``braid = 0`` it is *perfect* (spanning tree over
    rooms: exactly one simple path between any two free cells); ``braid > 0``
    removes dead ends to create loops (see :func:`braid_maze`).

    Parameters
    ----------
    algorithm : ``"dfs"`` (randomised depth-first search / recursive
        backtracker; long winding corridors, few branches) or ``"prim"``
        (randomised Prim's; many short dead ends).
    braid : probability that each dead end is opened into a neighbouring room.
        ``0`` leaves the maze perfect (and consumes no extra randomness, so
        existing seeds give identical layouts); ``1`` removes every dead end.
    """
    _check_dims(width, height)
    grid = np.ones((width, height), dtype=np.int8)  # fresh grid every call
    sx = 1 + 2 * int(rng.integers((width - 1) // 2))
    sy = 1 + 2 * int(rng.integers((height - 1) // 2))
    grid[sx, sy] = FREE

    def unvisited_neighbours(x, y):
        out = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[nx, ny] == WALL:
                out.append((nx, ny))
        return out

    if algorithm == "dfs":
        stack = [(sx, sy)]
        while stack:
            x, y = stack[-1]
            nbrs = unvisited_neighbours(x, y)
            if not nbrs:
                stack.pop()
                continue
            nx, ny = nbrs[int(rng.integers(len(nbrs)))]
            grid[(x + nx) // 2, (y + ny) // 2] = FREE
            grid[nx, ny] = FREE
            stack.append((nx, ny))
    elif algorithm == "prim":
        frontier = [(sx, sy, nx, ny) for nx, ny in unvisited_neighbours(sx, sy)]
        while frontier:
            i = int(rng.integers(len(frontier)))
            frontier[i], frontier[-1] = frontier[-1], frontier[i]
            x, y, nx, ny = frontier.pop()
            if grid[nx, ny] == WALL:
                grid[(x + nx) // 2, (y + ny) // 2] = FREE
                grid[nx, ny] = FREE
                frontier.extend((nx, ny, ax, ay) for ax, ay in unvisited_neighbours(nx, ny))
    else:
        raise ValueError(f"Unknown maze algorithm: {algorithm!r}")
    if braid > 0:
        braid_maze(grid, rng, braid)
    return grid


def braid_maze(grid: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    """In place: for every dead end (free cell with exactly one free neighbour),
    with probability ``p`` knock out one wall that separates it from another
    room. The result has cycles, so wall following no longer guarantees
    reaching every cell. Rooms stay connected, and the border stays wall."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"braid must be in [0, 1], got {p}")
    w, h = grid.shape
    dead_ends = [c for c in free_cells(grid)
                 if sum(grid[c[0] + dx, c[1] + dy] == FREE for dx, dy in ACTION_DELTAS) == 1]
    for x, y in dead_ends:
        if rng.random() >= p:
            continue
        # Candidate walls: adjacent wall with a free cell directly behind it
        # (inside the border), i.e. a wall between two rooms/corridors.
        cands = [(x + dx, y + dy) for dx, dy in ACTION_DELTAS
                 if grid[x + dx, y + dy] == WALL
                 and 0 < x + 2 * dx < w - 1 and 0 < y + 2 * dy < h - 1
                 and grid[x + 2 * dx, y + 2 * dy] == FREE]
        if cands:
            wx, wy = cands[int(rng.integers(len(cands)))]
            grid[wx, wy] = FREE
    return grid


def free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in c) for c in np.argwhere(grid == FREE)]


def bfs_distances(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Geodesic (shortest-path) distance from ``source`` to every cell.
    Walls and unreachable cells get ``-1``."""
    w, h = grid.shape
    dist = np.full((w, h), -1, dtype=np.int32)
    dist[source] = 0
    q = deque([source])
    while q:
        x, y = q.popleft()
        for dx, dy in ACTION_DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and grid[nx, ny] == FREE and dist[nx, ny] < 0:
                dist[nx, ny] = dist[x, y] + 1
                q.append((nx, ny))
    return dist


def place_entities(
    grid: np.ndarray,
    rng: np.random.Generator,
    min_distance: int = 3,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample (nest, food) uniformly among free-cell pairs whose *geodesic*
    distance is at least ``min_distance`` (falls back to the farthest cell if
    no cell qualifies)."""
    cells = free_cells(grid)
    nest = cells[int(rng.integers(len(cells)))]
    dist = bfs_distances(grid, nest)
    valid = [c for c in cells if dist[c] >= min_distance]
    if not valid:
        far = int(dist.max())
        valid = [c for c in cells if dist[c] == far]
    food = valid[int(rng.integers(len(valid)))]
    return nest, food


# --------------------------------------------------------------------------
# Train / test partition of the layout space.
#
# Randomised DFS is far from uniform over spanning trees, so the same layout
# can be produced by many seeds. Disjoint *seed* ranges therefore do not
# guarantee disjoint *layouts*. Instead, every layout is assigned to a split
# by a hash of its bytes; training only ever uses "train" layouts and held-out
# evaluation only "test" layouts.
# --------------------------------------------------------------------------
TEST_FRACTION_MOD = 10  # 1 in 10 layouts is reserved for testing


def layout_split(grid: np.ndarray) -> str:
    h = hashlib.blake2b(np.ascontiguousarray(grid).tobytes(), digest_size=8).digest()
    return "test" if int.from_bytes(h, "little") % TEST_FRACTION_MOD == 0 else "train"


def maze_from_seed(width: int, height: int, seed: int, algorithm: str = "dfs",
                   braid: float = 0.0) -> np.ndarray:
    return generate_maze(width, height, np.random.default_rng(seed), algorithm, braid)


@lru_cache(maxsize=64)
def seeds_in_split(n: int, base: int, split: str, width: int = 11, height: int = 11,
                   algorithm: str = "dfs", braid: float = 0.0) -> tuple[int, ...]:
    """First ``n`` seeds >= ``base`` whose layout falls in ``split``."""
    out, seed = [], base
    while len(out) < n:
        if split == "any" or layout_split(maze_from_seed(width, height, seed, algorithm, braid)) == split:
            out.append(seed)
        seed += 1
    return tuple(out)
