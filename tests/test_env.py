import numpy as np
import pytest

from ants_rl.baselines import OraclePolicy, WallFollowerPolicy
from ants_rl.env import AntMazeEnv, resolve_reward
from ants_rl.evaluation import FrameStacker, evaluate_baseline, heldout_episodes, summarise
from ants_rl.maze import bfs_distances, free_cells, generate_maze, layout_split


@pytest.mark.parametrize("algo", ["dfs", "prim"])
@pytest.mark.parametrize("size", [5, 11, 21])
def test_maze_is_perfect(algo, size):
    """Spanning tree over rooms: connected, #free = 2R-1, #edges = #nodes-1."""
    g = generate_maze(size, size, np.random.default_rng(0), algo)
    cells = free_cells(g)
    rooms = ((size - 1) // 2) ** 2
    assert len(cells) == 2 * rooms - 1
    d = bfs_distances(g, cells[0])
    assert all(d[c] >= 0 for c in cells)  # connected
    edges = sum(
        1 for (x, y) in cells for (dx, dy) in ((1, 0), (0, 1))
        if g[x + dx, y + dy] == 0
    )
    assert edges == len(cells) - 1  # acyclic
    assert g[0, :].all() and g[-1, :].all() and g[:, 0].all() and g[:, -1].all()


def test_maze_is_function_of_seed_and_varies():
    a = generate_maze(11, 11, np.random.default_rng(1))
    b = generate_maze(11, 11, np.random.default_rng(1))
    assert np.array_equal(a, b)
    layouts = {generate_maze(11, 11, np.random.default_rng(s)).tobytes() for s in range(50)}
    assert len(layouts) == 50  # the v0 bug produced a single layout


def test_procedural_env_regenerates_layout():
    env = AntMazeEnv(n_train_mazes=0)
    env.reset(seed=0)
    layouts = set()
    for _ in range(30):
        env.reset()
        layouts.add(env.grid.tobytes())
    assert len(layouts) == 30


def test_pool_env_uses_exactly_n_layouts():
    env = AntMazeEnv(n_train_mazes=3)
    env.reset(seed=0)
    seeds = {env.reset()[1]["maze_seed"] for _ in range(200)}
    assert len(seeds) == 3 and seeds == set(env._pool)


def test_train_and_heldout_layouts_disjoint():
    held = {AntMazeEnv()._maze(s).tobytes() for s, _ in heldout_episodes(200)}
    assert all(layout_split(np.frombuffer(b, dtype=np.int8).reshape(11, 11)) == "test" for b in held)
    for pool in (0, 64):
        env = AntMazeEnv(n_train_mazes=pool)
        env.reset(seed=1)
        for _ in range(300):
            env.reset()
            assert layout_split(env.grid) == "train"
            assert env.grid.tobytes() not in held


def test_seeding_reproducible():
    def rollout(seed):
        env = AntMazeEnv()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(0)
        traj = [obs]
        for _ in range(50):
            obs, *_ = env.step(int(rng.integers(4)))
            traj.append(obs)
        return np.stack(traj), env.grid.copy()
    a, ga = rollout(7)
    b, gb = rollout(7)
    assert np.array_equal(a, b) and np.array_equal(ga, gb)


def test_options_pin_episode():
    e1, e2 = AntMazeEnv(), AntMazeEnv()
    o1, _ = e1.reset(seed=1, options={"maze_seed": 123, "placement_seed": 9})
    o2, _ = e2.reset(seed=2, options={"maze_seed": 123, "placement_seed": 9})
    assert np.array_equal(o1, o2) and e1.food_pos == e2.food_pos


def test_observation_matches_grid():
    env = AntMazeEnv(obs_window=2)
    env.reset(seed=3)
    for _ in range(40):
        obs, *_ = env.step(env.action_space.sample())
        ax, ay = env.agent_pos
        for i in range(5):
            for j in range(5):
                x, y = ax + i - 2, ay + j - 2
                inside = 0 <= x < env.width and 0 <= y < env.height
                assert obs[0, i, j] == (env.grid[x, y] if inside else 1)
                assert obs[1, i, j] == float((x, y) == env.food_pos)
                assert obs[2, i, j] == float((x, y) == env.nest_pos)
        assert obs[0, 2, 2] == 0  # agent never stands in a wall


def test_min_food_distance_is_geodesic():
    env = AntMazeEnv(min_food_dist=6)
    env.reset(seed=0)
    for _ in range(50):
        env.reset()
        assert env.optimal_steps >= 6


def test_pbrs_telescopes():
    """Sum of shaping terms along any path = gamma^T Phi(s_T) - Phi(s_0)
    (discounted), so the discounted shaped return of the oracle differs from
    the unshaped one exactly by -Phi(s_0)."""
    gamma = 0.97
    env = AntMazeEnv(reward="pbrs", gamma=gamma)
    sparse = AntMazeEnv(reward="sparse", gamma=gamma)
    for s in range(20):
        opts = {"maze_seed": 500 + s, "placement_seed": s}
        env.reset(options=opts)
        sparse.reset(options=opts)
        phi0 = -env.dist_to_food[env.nest_pos] / env.max_dist
        rng = np.random.default_rng(s)
        G_shaped = G_sparse = 0.0
        for t in range(env.max_steps):
            a = int(rng.integers(4))
            _, r1, term, trunc, _ = env.step(a)
            _, r2, *_ = sparse.step(a)
            G_shaped += gamma**t * r1
            G_sparse += gamma**t * r2
            if term or trunc:
                break
        phiT = 0.0 if term else -env.dist_to_food[env.agent_pos] / env.max_dist
        assert G_shaped - G_sparse == pytest.approx(gamma ** (t + 1) * phiT - phi0, abs=1e-9)


def test_reward_presets():
    assert resolve_reward("legacy")["success"] == 10.0
    assert resolve_reward({"preset": "pbrs", "shaping_weight": 0.5})["shaping_weight"] == 0.5
    with pytest.raises(ValueError):
        resolve_reward({"preset": "pbrs", "bogus": 1})


def test_baselines():
    eps = heldout_episodes(30)
    s_or = summarise(evaluate_baseline(OraclePolicy(), {}, eps))
    assert s_or["success_rate"] == 1.0 and s_or["spl"] == pytest.approx(1.0)
    s_wf = summarise(evaluate_baseline(WallFollowerPolicy(), {}, eps))
    assert s_wf["success_rate"] == 1.0 and s_wf["wall_hit_rate"] == 0.0


def test_frame_stacker_matches_sb3():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    venv = VecFrameStack(DummyVecEnv([lambda: AntMazeEnv()]), n_stack=4, channels_order="first")
    venv.seed(5)
    o = venv.reset()
    raw = venv.venv.envs[0]
    st = FrameStacker(1, (3, 7, 7), 4)
    st.reset(0, raw._get_obs())
    assert np.array_equal(o[0], st.buf[0])
    for a in [0, 1, 2, 3, 1, 1, 2]:
        o, _, done, _ = venv.step(np.array([a]))
        if done[0]:
            break
        st.push(0, raw._get_obs())
        assert np.array_equal(o[0], st.buf[0])
