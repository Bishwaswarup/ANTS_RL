#!/usr/bin/env python
"""Train one agent and run the standard evaluation.

Examples
--------
    python scripts/train.py                                   # defaults
    python scripts/train.py --set agent.memory=lstm env.n_train_mazes=8 seed=3
    python scripts/train.py --config my.yaml --run-dir runs/debug
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
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(cfg)
    print(f"[train] run_dir = {run_dir}")
    summary = train(cfg, run_dir)
    print(json.dumps({k: summary[k] for k in ("heldout", "train_dist")}, indent=1))
    print(f"[train] done in {summary['train_time_s']:.0f}s")


if __name__ == "__main__":
    main()
