import numpy as np
import pytest

from ants_rl.baselines import WallFollowerPolicy
from ants_rl.evaluation import evaluate_baseline, heldout_episodes
from ants_rl.maze import ACTION_DELTAS, FREE, bfs_distances, free_cells, generate_maze


def n_dead_ends(g):
    return sum(1 for c in free_cells(g)
               if sum(g[c[0] + dx, c[1] + dy] == FREE for dx, dy in ACTION_DELTAS) == 1)


def n_edges(g):
    return sum(1 for (x, y) in free_cells(g) for dx, dy in ((1, 0), (0, 1)) if g[x + dx, y + dy] == FREE)


def test_braid_zero_is_identical_to_perfect_maze():
    for s in range(10):
        a = generate_maze(11, 11, np.random.default_rng(s))
        b = generate_maze(11, 11, np.random.default_rng(s), braid=0.0)
        assert np.array_equal(a, b)


@pytest.mark.parametrize("size", [11, 15])
def test_braid_one_removes_dead_ends_and_creates_loops(size):
    for s in range(10):
        g = generate_maze(size, size, np.random.default_rng(s), braid=1.0)
        cells = free_cells(g)
        d = bfs_distances(g, cells[0])
        assert all(d[c] >= 0 for c in cells)                    # still connected
        assert n_edges(g) > len(cells) - 1                     # cycles exist
        assert g[0, :].all() and g[-1, :].all() and g[:, 0].all() and g[:, -1].all()
        assert n_dead_ends(g) < n_dead_ends(generate_maze(size, size, np.random.default_rng(s)))


def test_braid_is_a_function_of_the_seed():
    a = generate_maze(11, 11, np.random.default_rng(3), braid=0.5)
    b = generate_maze(11, 11, np.random.default_rng(3), braid=0.5)
    assert np.array_equal(a, b)


def test_wall_follower_fails_on_some_braided_mazes_but_never_on_perfect_ones():
    cfg = dict(width=11, height=11)
    perfect = evaluate_baseline(WallFollowerPolicy(), dict(cfg, braid=0.0),
                                heldout_episodes(100, braid=0.0))
    braided = evaluate_baseline(WallFollowerPolicy(), dict(cfg, braid=1.0),
                                heldout_episodes(100, braid=1.0))
    assert all(r.success for r in perfect)
    assert not all(r.success for r in braided)
