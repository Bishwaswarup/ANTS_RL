"""
Rollouts for visualisation and frame drawing (no pygame / matplotlib needed).

``run_ants`` puts ``n_ants`` copies of one policy at the nest of the *same*
episode and records every ant's path. The ants do not interact (they cannot
see each other); with ``n_ants > 1`` the picture shows the spread of the
policy's behaviour, a "swarm" of independent foragers, rather than a colony
that coordinates.

``draw_frame`` turns the recorded paths into an RGB image at time ``t``:
walls dark, the field of view of a single ant tinted blue, cells visited so
far shaded from pale to deep red by how many ant-visits they received, nest
brown, food green, ants red (an ant that reached the food turns gold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import AntMazeEnv
from .evaluation import FrameStacker

C_WALL = np.array((40, 40, 40))
C_FREE = np.array((245, 245, 245))
C_VIEW = np.array((214, 228, 250))
C_TRAIL_LO = np.array((255, 214, 214))
C_TRAIL_HI = np.array((178, 24, 43))
C_NEST = np.array((139, 69, 19))
C_FOOD = np.array((50, 170, 50))
C_ANT = np.array((210, 30, 60))
C_DONE = np.array((240, 180, 0))


@dataclass
class Rollout:
    grid: np.ndarray
    nest: tuple
    food: tuple
    obs_window: int
    optimal_steps: int
    paths: list = field(default_factory=list)      # per ant: list of (x, y)
    success: list = field(default_factory=list)    # per ant: bool

    @property
    def length(self) -> int:
        return max(len(p) for p in self.paths)


def run_ants(env_cfg: dict, spec: tuple[int, int], n_ants: int = 1, model=None, cfg=None,
             baseline_cls=None, deterministic: bool = False, seed: int = 0) -> Rollout:
    """Roll out ``n_ants`` independent ants on episode ``spec`` = (maze_seed, placement_seed)."""
    envs = [AntMazeEnv(**env_cfg) for _ in range(n_ants)]
    opts = {"maze_seed": spec[0], "placement_seed": spec[1]}
    obs = [e.reset(options=opts)[0] for e in envs]
    e0 = envs[0]
    ro = Rollout(e0.grid.copy(), e0.nest_pos, e0.food_pos, e0.obs_window, e0.optimal_steps,
                 paths=[[e.agent_pos] for e in envs], success=[False] * n_ants)

    memory = cfg["agent"]["memory"] if cfg else None
    stacker = None
    if model is not None:
        import torch
        torch.manual_seed(seed)
        n_stack = cfg["agent"]["n_stack"] if memory == "stack" else 1
        stacker = FrameStacker(n_ants, obs[0].shape, n_stack)
        for i, o in enumerate(obs):
            stacker.reset(i, o)
    policies = []
    if baseline_cls is not None:
        for i, e in enumerate(envs):
            try:
                p = baseline_cls(seed=seed + i)
            except TypeError:
                p = baseline_cls()
            p.reset(e)
            policies.append(p)

    active = [True] * n_ants
    state, start = None, np.ones(n_ants, dtype=bool)
    while any(active):
        if model is not None:
            if memory == "lstm":
                actions, state = model.predict(stacker.buf, state=state, episode_start=start,
                                               deterministic=deterministic)
            else:
                actions, _ = model.predict(stacker.buf, deterministic=deterministic)
            start[:] = False
        else:
            actions = [p.act(e) for p, e in zip(policies, envs)]
        for i, e in enumerate(envs):
            if not active[i]:
                continue
            o, _, term, trunc, _ = e.step(int(actions[i]))
            if stacker is not None:
                stacker.push(i, o)
            ro.paths[i].append(e.agent_pos)
            if term or trunc:
                active[i] = False
                ro.success[i] = bool(term)
    return ro


def _disc(img, cx, cy, r, color):
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = color


def draw_frame(ro: Rollout, t: int, cell: int = 20) -> np.ndarray:
    """RGB image (H, W, 3) of every ant's state after ``t`` steps."""
    w, h = ro.grid.shape
    counts = np.zeros((w, h))
    for p in ro.paths:
        for (x, y) in p[: t + 1]:
            counts[x, y] += 1
    heat = np.log1p(counts) / max(np.log1p(counts.max()), 1e-9)
    single = len(ro.paths) == 1
    img = np.zeros((h * cell, w * cell, 3), dtype=np.uint8)
    pos = [p[min(t, len(p) - 1)] for p in ro.paths]
    ax, ay = pos[0]
    k = ro.obs_window
    for x in range(w):
        for y in range(h):
            if ro.grid[x, y]:
                c = C_WALL
            elif counts[x, y] > 0:
                c = C_TRAIL_LO + heat[x, y] * (C_TRAIL_HI - C_TRAIL_LO) * (0.0 if single else 1.0)
            elif single and abs(x - ax) <= k and abs(y - ay) <= k:
                c = C_VIEW
            else:
                c = C_FREE
            if (x, y) == ro.nest:
                c = C_NEST
            if (x, y) == ro.food:
                c = C_FOOD
            img[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell] = c
    r = max(2, cell // 3 if single else cell // 4)
    jitter = np.random.default_rng(0).uniform(-cell / 5, cell / 5, size=(len(pos), 2)) if not single \
        else np.zeros((1, 2))
    for i, (x, y) in enumerate(pos):
        finished = t >= len(ro.paths[i]) - 1 and ro.success[i]
        _disc(img, x * cell + cell // 2 + jitter[i, 0], y * cell + cell // 2 + jitter[i, 1], r,
              C_DONE if finished else C_ANT)
    return img


def caption(img: np.ndarray, lines: list[str], height: int = 34) -> np.ndarray:
    """Add white header rows with text above ``img``."""
    from PIL import Image, ImageDraw
    out = Image.new("RGB", (img.shape[1], img.shape[0] + height), (255, 255, 255))
    out.paste(Image.fromarray(img), (0, height))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((4, 2 + 15 * i), line, fill=(20, 20, 20))
    return np.asarray(out)


def hstack(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    parts = []
    for k, im in enumerate(images):
        pad = np.full((h, im.shape[1], 3), 255, dtype=np.uint8)
        pad[: im.shape[0]] = im
        parts.append(pad)
        if k < len(images) - 1:
            parts.append(np.full((h, gap, 3), 255, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def vstack(images: list[np.ndarray]) -> np.ndarray:
    w = max(i.shape[1] for i in images)
    parts = []
    for im in images:
        pad = np.full((im.shape[0], w, 3), 255, dtype=np.uint8)
        pad[:, : im.shape[1]] = im
        parts.append(pad)
    return np.concatenate(parts, axis=0)


def save_gif(frames: list[np.ndarray], path: str, fps: int = 10, hold: int = 0) -> None:
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames] + [Image.fromarray(frames[-1])] * hold
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0,
                 optimize=True)
