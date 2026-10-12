#!/usr/bin/env python
"""Watch a trained ant (or a baseline, or a "swarm" of copies) on held-out mazes.

    python scripts/render.py --run runs/gate/3M --live --episodes 5
    python scripts/render.py --run runs/gate/3M --gif ant.gif --episodes 3
    python scripts/render.py --run runs/gate/3M --ants 20 --gif swarm.gif   # 20 independent ants
    python scripts/render.py --run runs/gate/3M --size 21 --live            # bigger, unseen maze
    python scripts/render.py --baseline wall_follower --live

Walls are dark, the ant's field of view is blue (single ant), visited cells
are pink (single ant) or shaded by how many ants passed (several ants), the
nest is brown, the food green, ants red, and an ant that found the food turns
gold. The maze drawn is exactly the maze the policy acts in. With --ants N the
N ants are copies of the same policy that cannot see each other: the picture
shows how varied the policy's behaviour is, not coordination.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ants_rl.baselines import BASELINES  # noqa: E402
from ants_rl.config import load_config, load_yaml  # noqa: E402
from ants_rl.evaluation import heldout_episodes  # noqa: E402
from ants_rl.training import load_model  # noqa: E402
from ants_rl.viz import caption, draw_frame, run_ants, save_gif  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run directory (uses model.zip)")
    ap.add_argument("--model", default=None, help="other policy file in the run, e.g. snapshots/step_00050000.zip")
    ap.add_argument("--baseline", choices=list(BASELINES), default=None)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0, help="which held-out episodes to show")
    ap.add_argument("--ants", type=int, default=1, help="independent copies of the policy")
    ap.add_argument("--size", type=int, default=None, help="maze size override (odd)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--cell", type=int, default=24, help="pixels per cell")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--live", action="store_true", help="pygame window (pip install pygame)")
    args = ap.parse_args()

    model, cfg, base_cls = None, None, None
    if args.run:
        cfg = load_config(overrides=load_yaml(Path(args.run) / "config.yaml"))
        path = Path(args.run) / args.model if args.model else None
        model = load_model(args.run, cfg, path=path)
        env_cfg = dict(cfg["env"])
        who = f"{Path(args.run).name}" + (f" [{args.model}]" if args.model else "")
    elif args.baseline:
        base_cls = BASELINES[args.baseline]
        env_cfg = dict(load_config()["env"])
        who = args.baseline
    else:
        ap.error("give --run or --baseline")
    if args.size:
        env_cfg.update(width=args.size, height=args.size, max_steps=None)
    env_cfg["n_train_mazes"] = 0

    mode = "deterministic" if args.deterministic else "stochastic"
    frames = []
    specs = heldout_episodes(args.episodes, offset=args.offset, width=env_cfg["width"],
                             height=env_cfg["height"], algorithm=env_cfg.get("maze_algorithm", "dfs"),
                             braid=env_cfg.get("braid", 0.0))
    for n, spec in enumerate(specs):
        ro = run_ants(env_cfg, spec, args.ants, model=model, cfg=cfg, baseline_cls=base_cls,
                      deterministic=args.deterministic, seed=n)
        found = sum(ro.success)
        lengths = [len(p) - 1 for p, s in zip(ro.paths, ro.success) if s]
        print(f"maze {spec[0]}: {found}/{args.ants} ants found food"
              + (f", fastest {min(lengths)} steps" if lengths else "")
              + f" (shortest path {ro.optimal_steps})")
        for t in range(ro.length):
            done = sum(1 for p, s in zip(ro.paths, ro.success) if s and len(p) - 1 <= t)
            frames.append(caption(draw_frame(ro, t, args.cell), [
                f"{who}  ({mode})",
                f"maze {n + 1}/{len(specs)}  step {t}  found food: {done}/{args.ants}"]))
        frames += [frames[-1]] * args.fps  # pause between episodes

    if args.gif:
        save_gif(frames, args.gif, fps=args.fps)
        print(f"wrote {args.gif} ({len(frames)} frames)")
    if args.live:
        import time

        import pygame
        pygame.init()
        h, w = frames[0].shape[:2]
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("ANTS-RL")
        for f in frames:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    return
            screen.blit(pygame.surfarray.make_surface(f.swapaxes(0, 1)), (0, 0))
            pygame.display.flip()
            time.sleep(1 / args.fps)
        pygame.quit()


if __name__ == "__main__":
    main()
