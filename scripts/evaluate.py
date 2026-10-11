#!/usr/bin/env python
"""Re-evaluate trained runs, or evaluate the non-learned baselines.

Examples
--------
    # re-run the standard evaluation for a trained run (overwrites eval.json)
    python scripts/evaluate.py --run runs/main/stack4_mazes0/seed0

    # baselines on the same held-out episodes -> results/baselines.json
    python scripts/evaluate.py --baselines --out results/baselines.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ants_rl.baselines import BASELINES  # noqa: E402
from ants_rl.config import load_config, load_yaml  # noqa: E402
from ants_rl.evaluation import (evaluate_baseline, heldout_episodes, save_results,  # noqa: E402
                                summarise)
from ants_rl.training import (full_evaluation, load_model, save_episode_arrays)  # noqa: E402


def eval_run(run_dir: Path):
    cfg = load_config(overrides=load_yaml(run_dir / "config.yaml"))
    model = load_model(run_dir, cfg)
    summary, episodes = full_evaluation(model, cfg)
    old = run_dir / "eval.json"
    if old.exists():
        prev = json.loads(old.read_text())
        for k in ("train_time_s", "timesteps"):
            if k in prev:
                summary[k] = prev[k]
    save_results(old, summary)
    save_episode_arrays(run_dir / "episodes.npz", episodes)
    print(run_dir, json.dumps(summary["heldout"], indent=1))


def eval_baselines(cfg_path, out: Path):
    cfg = load_config(cfg_path)
    e = cfg["eval"]
    sizes = [cfg["env"]["width"]] + list(e.get("transfer_sizes") or [])
    payload = {}
    for name, cls in BASELINES.items():
        payload[name] = {}
        for size in sizes:
            ecfg = dict(cfg["env"], width=size, height=size, n_train_mazes=0)
            if size != cfg["env"]["width"]:
                ecfg["max_steps"] = None
            res = evaluate_baseline(cls(), ecfg, heldout_episodes(e["n_episodes"], width=size, height=size,
                                                                  algorithm=ecfg["maze_algorithm"]),
                                    gamma=cfg["ppo"]["gamma"])
            payload[name][str(size)] = summarise(res)
            s = payload[name][str(size)]
            print(f"{name:14s} {size:3d}x{size:<3d} success={s['success_rate']:.3f} spl={s['spl']:.3f}")
    save_results(out, payload)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=[], help="run directories")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()
    for r in args.run:
        eval_run(Path(r))
    if args.baselines:
        eval_baselines(args.config, Path(args.out))
    if not args.run and not args.baselines:
        ap.print_help()


if __name__ == "__main__":
    main()
