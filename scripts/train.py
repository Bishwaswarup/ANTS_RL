#!/usr/bin/env python
"""Train one agent and run the standard evaluation.

Examples
--------
    python scripts/train.py                                   # defaults
    python scripts/train.py --set agent.memory=lstm env.n_train_mazes=8 seed=3
    python scripts/train.py --config my.yaml --run-dir runs/debug

    # one run at a time, each maze in its own process, stop after 3 h
    # (re-run the same command to continue from the checkpoint)
    python scripts/train.py --subproc --max-hours 3 --run-dir runs/demo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from ants_rl.config import load_config  # noqa: E402
from ants_rl.training import train  # noqa: E402


def default_run_dir(cfg) -> Path:
    a, e = cfg["agent"], cfg["env"]
    mem = f"stack{a['n_stack']}" if a["memory"] == "stack" else a["memory"]
    tag = f"{mem}_mazes{e['n_train_mazes']}_{e['reward']['preset']}_w{e['obs_window']}"
    return Path(cfg["output"]["root"]) / "single" / tag / f"seed{cfg['seed']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML merged over configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    ap.add_argument("--subproc", action="store_true",
                    help="run each maze in its own process (= --set ppo.vec_env=subproc)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop after this many hours (checkpoint saved; re-run to continue)")
    ap.add_argument("--force", action="store_true", help="retrain even if eval.json exists")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    overrides = list(args.set) + (["ppo.vec_env=subproc"] if args.subproc else [])
    cfg = load_config(args.config, overrides)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(cfg)
    print(f"[train] run_dir = {run_dir}")
    if (run_dir / "eval.json").exists() and not args.force:
        print("[train] already finished (eval.json exists); use --force to retrain")
        return
    summary = train(cfg, run_dir, max_hours=args.max_hours)
    if summary is None:
        sys.exit(3)  # paused by the time budget
    print(json.dumps({k: summary[k] for k in ("heldout", "train_dist")}, indent=1))
    print(f"[train] done; training time {summary['train_time_s'] / 60:.0f} min "
          f"over {summary['sessions']} session(s)")


if __name__ == "__main__":
    main()
