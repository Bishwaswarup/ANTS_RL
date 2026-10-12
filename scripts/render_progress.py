#!/usr/bin/env python
"""Watch an ant learn: the same unseen maze, solved by the policy at several
points in training, side by side, above the run's learning curve.

Training saves policy copies at the steps in ``output.snapshots`` (default
0, 50k, 200k, 500k, 1M) into <run>/snapshots/; the final policy is model.zip.

    python scripts/render_progress.py --run runs/demo --gif progress.gif
    python scripts/render_progress.py --run runs/demo --ants 20 --gif swarm_progress.gif
    python scripts/render_progress.py --run runs/demo --png paper/figures/progress.png --offset 3

--ants N runs N independent copies of each policy (they cannot see each
other), which makes the spread of behaviour at each stage visible: early on
the ants wander and pile up against walls, later most of them stream to the food.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ants_rl.baselines import RandomPolicy, WallFollowerPolicy  # noqa: E402
from ants_rl.config import load_config, load_yaml  # noqa: E402
from ants_rl.evaluation import evaluate_baseline, heldout_episodes, summarise  # noqa: E402
from ants_rl.training import load_model  # noqa: E402
from ants_rl.viz import caption, draw_frame, hstack, run_ants, save_gif, vstack  # noqa: E402

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"


def fmt_steps(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1_000_000 else (f"{n // 1000}k" if n >= 1000 else str(n))


def policies(run: Path, max_panels: int) -> list[tuple[int, Path]]:
    snaps = []
    for p in sorted((run / "snapshots").glob("step_*.zip")):
        m = re.match(r"step_(\d+)\.zip", p.name)
        if m:
            snaps.append((int(m.group(1)), p))
    if (run / "model.zip").exists():
        steps = json.loads((run / "eval.json").read_text())["timesteps"] if (run / "eval.json").exists() \
            else (snaps[-1][0] + 1 if snaps else 0)
        snaps.append((int(steps), run / "model.zip"))
    if not snaps:
        sys.exit(f"no snapshots/ or model.zip in {run}; train with output.snapshots set")
    if len(snaps) > max_panels:  # keep first and last, spread the rest
        idx = np.unique(np.linspace(0, len(snaps) - 1, max_panels).round().astype(int))
        snaps = [snaps[i] for i in idx]
    return snaps


def curve_value(curves, step: int, col: str):
    if curves is None or col not in curves:
        return None
    i = int(np.argmin(np.abs(curves["timesteps"].values - step)))
    return float(curves[col].values[i])


def reference_levels(cfg: dict, env_cfg: dict) -> dict:
    """Random walk and wall follower on the *same* held-out episodes the
    learning curve uses (eval.curve_episodes), so the levels are comparable."""
    eps = heldout_episodes(cfg["eval"]["curve_episodes"], width=env_cfg["width"],
                           height=env_cfg["height"], algorithm=env_cfg.get("maze_algorithm", "dfs"),
                           braid=env_cfg.get("braid", 0.0))
    return {name: summarise(evaluate_baseline(pol, env_cfg, eps, gamma=cfg["ppo"]["gamma"]))["success_rate"]
            for name, pol in (("wall follower", WallFollowerPolicy()), ("random walk", RandomPolicy()))}


def curve_image(curves, marks, width_px: int, refs: dict) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, 2.4), dpi=dpi)
    x = curves["timesteps"].values / 1e6
    ax.plot(x, curves["stochastic_success"], color=BLUE, lw=2, label="success (stochastic)")
    if "deterministic_success" in curves:
        ax.plot(x, curves["deterministic_success"], color=ORANGE, lw=2, ls="--",
                label="success (deterministic)")
    for (name, v), ls in zip(refs.items(), ((0, (4, 2)), ":")):
        ax.axhline(v, color=MUTED, lw=1, ls=ls)
        ax.annotate(f"{name} ({100 * v:.0f}%)", (x.min(), v), xytext=(4, 3),
                    textcoords="offset points", ha="left", fontsize=7, color=MUTED)
    for k, (step, _) in enumerate(marks):
        ax.axvline(step / 1e6, color=MUTED, lw=0.8, alpha=0.6)
        ax.annotate(chr(65 + k), (step / 1e6, 1.02), ha="center", fontsize=8, color=MUTED,
                    annotation_clip=False)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
    ax.set_ylabel("held-out success", fontsize=8, color=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.yaxis.grid(True, color=GRID)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--offset", type=int, default=0, help="which held-out maze")
    ap.add_argument("--ants", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-panels", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=200, help="cap animation length")
    ap.add_argument("--cell", type=int, default=18)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--png", default=None, help="static figure: every path drawn in full")
    args = ap.parse_args()
    if not (args.gif or args.png):
        ap.error("give --gif and/or --png")

    run = Path(args.run)
    cfg = load_config(overrides=load_yaml(run / "config.yaml"))
    env_cfg = dict(cfg["env"], n_train_mazes=0)
    spec = heldout_episodes(args.offset + 1, width=env_cfg["width"], height=env_cfg["height"],
                            algorithm=env_cfg.get("maze_algorithm", "dfs"),
                            braid=env_cfg.get("braid", 0.0))[args.offset]
    curves = None
    if (run / "curves.csv").exists():
        import pandas as pd
        curves = pd.read_csv(run / "curves.csv")
    mode = "deterministic" if args.deterministic else "stochastic"

    marks = policies(run, args.max_panels)
    rollouts, labels = [], []
    for k, (step, path) in enumerate(marks):
        model = load_model(run, cfg, path=path)
        ro = run_ants(env_cfg, spec, args.ants, model=model, cfg=cfg,
                      deterministic=args.deterministic, seed=k)
        rollouts.append(ro)
        sr = curve_value(curves, step, f"{mode}_success")
        labels.append(f"{chr(65 + k)}: {fmt_steps(step)} steps"
                      + (f" | held-out {100 * sr:.0f}%" if sr is not None else ""))
        print(f"{labels[-1]:40s} on this maze: {sum(ro.success)}/{args.ants} found food")

    T = min(max(r.length for r in rollouts), args.max_steps + 1)

    def frame(t):
        panels = []
        for ro, lab in zip(rollouts, labels):
            done = sum(1 for p, s in zip(ro.paths, ro.success) if s and len(p) - 1 <= t)
            panels.append(caption(draw_frame(ro, t, args.cell),
                                  [lab, f"step {min(t, ro.length - 1)}  food: {done}/{len(ro.paths)}"]))
        return hstack(panels)

    top0 = frame(0)
    bottom = (curve_image(curves, marks, top0.shape[1], reference_levels(cfg, env_cfg))
              if curves is not None else None)
    title = caption(np.full((1, top0.shape[1], 3), 255, dtype=np.uint8),
                    [f"{run.name}: one unseen maze, policy at different stages of training "
                     f"({mode}, {args.ants} independent ant{'s' if args.ants > 1 else ''})"],
                    height=20)

    def compose(t):
        parts = [title, frame(t)] + ([bottom] if bottom is not None else [])
        return vstack(parts)

    if args.gif:
        frames = [compose(t) for t in range(T)]
        save_gif(frames, args.gif, fps=args.fps, hold=2 * args.fps)
        print(f"wrote {args.gif} ({len(frames)} frames)")
    if args.png:
        from PIL import Image
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(compose(max(r.length for r in rollouts) - 1)).save(args.png)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
