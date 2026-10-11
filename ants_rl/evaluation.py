"""
Evaluation on fixed, reproducible episode sets.

An *episode spec* is ``(maze_seed, placement_seed)``. Every method (learned or
baseline) is evaluated on exactly the same specs, so differences are paired.

Metrics per episode: success, steps, optimal_steps, SPL (Anderson et al.,
2018: ``S * l / max(p, l)``), wall hits, unique cells visited, return.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .env import EVAL_SEED_BASE, AntMazeEnv
from .maze import seeds_in_split

TRAIN_PLACEMENT_BASE = 5_000_000


def heldout_episodes(n: int, offset: int = 0, width: int = 11, height: int = 11,
                     algorithm: str = "dfs") -> list[tuple[int, int]]:
    """``n`` episodes on held-out ("test"-split) layouts. Seeds are distinct;
    layouts are distinct up to the (rare) case of two seeds giving one layout."""
    seeds = seeds_in_split(offset + n, EVAL_SEED_BASE, "test", width, height, algorithm)[offset:]
    return [(s, s) for s in seeds]


def train_episodes(env_cfg: dict, n: int) -> list[tuple[int, int]]:
    """Episodes drawn from the *training* layout distribution, with fresh
    nest/food placements. For procedural training this is just fresh layouts."""
    pool = env_cfg.get("n_train_mazes", 0)
    w, h = env_cfg.get("width", 11), env_cfg.get("height", 11)
    alg = env_cfg.get("maze_algorithm", "dfs")
    if pool > 0:
        seeds = seeds_in_split(pool, env_cfg.get("maze_seed_base", 0), "train", w, h, alg)
        return [(seeds[i % pool], TRAIN_PLACEMENT_BASE + i) for i in range(n)]
    seeds = seeds_in_split(n, 20_000_000, "train", w, h, alg)
    return [(s, TRAIN_PLACEMENT_BASE + i) for i, s in enumerate(seeds)]


@dataclass
class EpisodeResult:
    maze_seed: int
    placement_seed: int
    success: bool
    steps: int
    optimal_steps: int
    spl: float
    wall_hits: int
    unique_cells: int
    ret: float


def _result(spec, env: AntMazeEnv, success: bool, ret: float) -> EpisodeResult:
    l, p = env.optimal_steps, env.step_count
    return EpisodeResult(
        maze_seed=spec[0], placement_seed=spec[1], success=bool(success),
        steps=p, optimal_steps=l,
        spl=float(success) * l / max(p, l, 1),
        wall_hits=env.wall_hits, unique_cells=len(env.visit_counts), ret=float(ret),
    )


def summarise(results: list[EpisodeResult]) -> dict:
    if not results:
        return {}
    s = np.array([r.success for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=float)
    out = {
        "n_episodes": len(results),
        "success_rate": float(s.mean()),
        "spl": float(np.mean([r.spl for r in results])),
        "mean_steps": float(steps.mean()),
        "mean_steps_success": float(steps[s > 0].mean()) if s.any() else float("nan"),
        "wall_hit_rate": float(np.sum([r.wall_hits for r in results]) / max(steps.sum(), 1)),
        "mean_unique_cells": float(np.mean([r.unique_cells for r in results])),
        "mean_return": float(np.mean([r.ret for r in results])),
    }
    return out


# --------------------------------------------------------------------- stacking
class FrameStacker:
    """Replicates ``VecFrameStack(channels_order="first")`` for a batch of envs:
    oldest frame first, zero padding after reset."""

    def __init__(self, n_envs: int, obs_shape, n_stack: int):
        c = obs_shape[0]
        self.c = c
        self.buf = np.zeros((n_envs, c * n_stack, *obs_shape[1:]), dtype=np.float32)

    def reset(self, i: int, obs: np.ndarray):
        self.buf[i] = 0.0
        self.buf[i, -self.c:] = obs

    def push(self, i: int, obs: np.ndarray):
        self.buf[i] = np.roll(self.buf[i], -self.c, axis=0)
        self.buf[i, -self.c:] = obs


# ----------------------------------------------------------------- SB3 models
def evaluate_model(
    model,
    env_cfg: dict,
    episodes: list[tuple[int, int]],
    memory: str = "stack",
    n_stack: int = 4,
    deterministic: bool = True,
    n_envs: int = 32,
    gamma: float = 0.99,
) -> list[EpisodeResult]:
    """Run ``model`` (PPO or RecurrentPPO) on every episode spec, batched."""
    n_envs = max(1, min(n_envs, len(episodes)))
    envs = [AntMazeEnv(gamma=gamma, **env_cfg) for _ in range(n_envs)]
    stack = n_stack if memory == "stack" else 1
    stacker = FrameStacker(n_envs, envs[0].observation_space.shape, stack)
    queue = list(enumerate(episodes))[::-1]
    slot_spec: list = [None] * n_envs
    slot_ret = np.zeros(n_envs)
    results: list = [None] * len(episodes)
    episode_start = np.ones(n_envs, dtype=bool)
    lstm_state = None

    def start(i):
        if not queue:
            slot_spec[i] = None
            return
        idx, spec = queue.pop()
        obs, _ = envs[i].reset(options={"maze_seed": spec[0], "placement_seed": spec[1]})
        stacker.reset(i, obs)
        slot_spec[i] = (idx, spec)
        slot_ret[i] = 0.0
        episode_start[i] = True

    for i in range(n_envs):
        start(i)

    while any(s is not None for s in slot_spec):
        obs = stacker.buf
        if memory == "lstm":
            actions, lstm_state = model.predict(
                obs, state=lstm_state, episode_start=episode_start, deterministic=deterministic
            )
        else:
            actions, _ = model.predict(obs, deterministic=deterministic)
        episode_start[:] = False
        for i in range(n_envs):
            if slot_spec[i] is None:
                continue
            o, r, term, trunc, _ = envs[i].step(int(actions[i]))
            slot_ret[i] += r
            stacker.push(i, o)
            if term or trunc:
                idx, spec = slot_spec[i]
                results[idx] = _result(spec, envs[i], term, slot_ret[i])
                start(i)
    return results


# ------------------------------------------------------------------ baselines
def evaluate_baseline(policy, env_cfg: dict, episodes, gamma: float = 0.99) -> list[EpisodeResult]:
    env = AntMazeEnv(gamma=gamma, **env_cfg)
    out = []
    for spec in episodes:
        env.reset(options={"maze_seed": spec[0], "placement_seed": spec[1]})
        policy.reset(env)
        ret, done, term = 0.0, False, False
        while not done:
            _, r, term, trunc, _ = env.step(policy.act(env))
            ret += r
            done = term or trunc
        out.append(_result(spec, env, term, ret))
    return out


def save_results(path: str | Path, payload: dict) -> None:
    def enc(o):
        if isinstance(o, EpisodeResult):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=1, default=enc))
