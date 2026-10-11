#!/usr/bin/env python
"""Loops out-of-distribution: evaluate on *braided* held-out mazes.

A perfect maze has exactly one path between two cells, so the right-hand wall
follower always succeeds. Braiding opens a fraction of the dead ends and
creates loops; the wall follower can then circle an island forever. This
script evaluates the baselines and any trained runs on the same held-out
layouts at several braid levels.

Examples
--------
    python scripts/eval_braid.py --runs runs/main/memory=lstm__n_train_mazes=0/seed* \
        runs/main/memory=stack__n_train_mazes=0/seed*
    python scripts/eval_braid.py --levels 0 0.25 0.5 1.0 --n-episodes 500   # baselines only

Writes results/braid.json (baselines + per-run summaries per level, mode).
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from ants_rl.baselines import BASELINES  # noqa: E402
from ants_rl.config import load_config, load_yaml  # noqa: E402
from ants_rl.evaluation import (evaluate_baseline, evaluate_model, heldout_episodes,  # noqa: E402
                                save_results, summarise)
from ants_rl.training import load_model  # noqa: E402


def level_cfg(env_cfg: dict, braid: float) -> dict:
    return dict(env_cfg, braid=braid, n_train_mazes=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run dirs or glob patterns")
    ap.add_argument("--levels", nargs="*", type=float, default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--modes", nargs="*", default=["stochastic", "deterministic"])
    ap.add_argument("--out", default="results/braid.json")
    args = ap.parse_args()
    torch.set_num_threads(1)

    base = load_config()
    result: dict = {"levels": args.levels, "n_episodes": args.n_episodes,
                    "baselines": {}, "runs": {}}
    episode_sets = {}
    for lv in args.levels:
        ecfg = level_cfg(base["env"], lv)
        episode_sets[lv] = heldout_episodes(args.n_episodes, width=ecfg["width"], height=ecfg["height"],
                                            algorithm=ecfg["maze_algorithm"], braid=lv)

    for name, cls in BASELINES.items():
        result["baselines"][name] = {}
        for lv in args.levels:
            s = summarise(evaluate_baseline(cls(), level_cfg(base["env"], lv), episode_sets[lv],
                                            gamma=base["ppo"]["gamma"]))
            result["baselines"][name][str(lv)] = s
            print(f"{name:14s} braid={lv:<4} success={s['success_rate']:.3f} spl={s['spl']:.3f}")

    run_dirs = sorted({Path(p) for pat in args.runs for p in glob.glob(pat)})
    for rd in run_dirs:
        if not (rd / "model.zip").exists():
            print(f"skip {rd}: no model.zip")
            continue
        cfg = load_config(overrides=load_yaml(rd / "config.yaml"))
        model = load_model(rd, cfg)
        a = cfg["agent"]
        out = result["runs"][str(rd)] = {"memory": a["memory"], "seed": cfg["seed"],
                                         "train_braid": cfg["env"].get("braid", 0.0),
                                         "n_train_mazes": cfg["env"]["n_train_mazes"],
                                         "lstm_hidden": a.get("lstm_hidden"), "levels": {}}
        for lv in args.levels:
            out["levels"][str(lv)] = {}
            for mode in args.modes:
                res = evaluate_model(model, level_cfg(cfg["env"], lv), episode_sets[lv],
                                     memory=a["memory"], n_stack=a["n_stack"],
                                     deterministic=(mode == "deterministic"),
                                     gamma=cfg["ppo"]["gamma"])
                s = summarise(res)
                out["levels"][str(lv)][mode] = s
                print(f"{rd} braid={lv:<4} {mode:13s} success={s['success_rate']:.3f} spl={s['spl']:.3f}")
    save_results(args.out, result)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
