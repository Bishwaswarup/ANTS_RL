#!/usr/bin/env python
"""Render a trained agent (or a baseline) on held-out mazes.

    python scripts/render.py --run runs/main/memory=stack__n_train_mazes=0/seed0 --gif out.gif
    python scripts/render.py --baseline wall_follower --gif wf.gif --episodes 3
    python scripts/render.py --run <run_dir> --live          # pygame window (pip install pygame)

The image shows walls (dark), the agent's current field of view (blue tint),
visited cells (pink), nest (brown), food (green) and the ant (red).
The rendered maze is exactly the maze the policy acts in.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ants_rl.baselines import BASELINES  # noqa: E402
from ants_rl.config import load_config, load_yaml  # noqa: E402
from ants_rl.env import AntMazeEnv  # noqa: E402
from ants_rl.evaluation import FrameStacker, heldout_episodes  # noqa: E402
from ants_rl.training import load_model  # noqa: E402


def frames_for_episode(env, spec, model=None, cfg=None, baseline=None, deterministic=False):
    obs, _ = env.reset(options={"maze_seed": spec[0], "placement_seed": spec[1]})
    trail = {env.agent_pos}
    frames = [env.render(trail=trail)]
    memory = cfg["agent"]["memory"] if cfg else None
    stacker = FrameStacker(1, obs.shape, cfg["agent"]["n_stack"] if memory == "stack" else 1) if model else None
    if stacker:
        stacker.reset(0, obs)
    state, start = None, np.ones(1, dtype=bool)
    if baseline:
        baseline.reset(env)
    done, term = False, False
    while not done:
        if baseline:
            a = baseline.act(env)
        elif memory == "lstm":
            act, state = model.predict(stacker.buf, state=state, episode_start=start,
                                       deterministic=deterministic)
            a, start = int(act[0]), np.zeros(1, dtype=bool)
        else:
            a = int(model.predict(stacker.buf, deterministic=deterministic)[0][0])
        obs, _, term, trunc, _ = env.step(a)
        if stacker:
            stacker.push(0, obs)
        trail.add(env.agent_pos)
        frames.append(env.render(trail=trail))
        done = term or trunc
    print(f"maze {spec[0]}: {'success' if term else 'timeout'} in {env.step_count} steps "
          f"(optimal {env.optimal_steps})")
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--baseline", choices=list(BASELINES), default=None)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0, help="which held-out episodes to show")
    ap.add_argument("--size", type=int, default=None, help="maze size override")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    model, cfg, baseline = None, None, None
    if args.run:
        cfg = load_config(overrides=load_yaml(Path(args.run) / "config.yaml"))
        model = load_model(args.run, cfg)
        env_cfg = dict(cfg["env"])
    elif args.baseline:
        baseline = BASELINES[args.baseline]()
        env_cfg = dict(load_config()["env"])
    else:
        ap.error("give --run or --baseline")
    if args.size:
        env_cfg.update(width=args.size, height=args.size, max_steps=None)
    env = AntMazeEnv(**env_cfg)

    frames = []
    for spec in heldout_episodes(args.episodes, offset=args.offset, width=env.width,
                                 height=env.height, algorithm=env.maze_algorithm):
        frames += frames_for_episode(env, spec, model, cfg, baseline, args.deterministic)
        frames += [frames[-1]] * args.fps  # pause between episodes

    if args.gif:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=1000 // args.fps, loop=0)
        print(f"wrote {args.gif} ({len(frames)} frames)")
    if args.live:
        import time

        import pygame
        pygame.init()
        h, w = frames[0].shape[:2]
        screen = pygame.display.set_mode((w, h))
        for f in frames:
            pygame.event.pump()
            screen.blit(pygame.surfarray.make_surface(f.swapaxes(0, 1)), (0, 0))
            pygame.display.flip()
            time.sleep(1 / args.fps)
        pygame.quit()


if __name__ == "__main__":
    main()
