#!/usr/bin/env python
"""Expand an experiment spec (configs/experiments/*.yaml) into runs and execute
them, optionally in parallel. Finished runs (eval.json present) are skipped,
so the command can be re-launched after interruption.

Examples
--------
    python scripts/run_experiments.py configs/experiments/main.yaml --dry-run
    python scripts/run_experiments.py configs/experiments/main.yaml --jobs 4
    python scripts/run_experiments.py configs/experiments/main.yaml --jobs 4 \
        --set ppo.total_timesteps=200000          # quick pilot
    python scripts/run_experiments.py configs/experiments/main.yaml --shard 0/4   # cluster arrays
"""

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ants_rl.config import deep_merge, load_config, parse_overrides, set_dotted  # noqa: E402


def short(key: str) -> str:
    return key.split(".")[-1]


def fmt(v) -> str:
    return str(v).replace("/", "-").replace(" ", "")


def expand(spec: dict, extra: dict) -> list[tuple[Path, dict]]:
    base = load_config(spec.get("base"))
    fixed: dict = {}
    for k, v in (spec.get("fixed") or {}).items():
        set_dotted(fixed, k, v)
    base = deep_merge(base, fixed)
    base = deep_merge(base, extra)
    grid = spec.get("grid") or {}
    keys = list(grid)
    variants = spec.get("variants") or [{}]  # optional list of extra override dicts
    runs = []
    root = Path(base["output"]["root"]) / spec["name"]
    for values in itertools.product(*(grid[k] for k in keys)):
        for var in variants:
            cfg = base
            tagparts = []
            for k, v in zip(keys, values):
                d: dict = {}
                set_dotted(d, k, v)
                cfg = deep_merge(cfg, d)
                tagparts.append(f"{short(k)}={fmt(v)}")
            if var:
                vd: dict = {}
                for k, v in var.items():
                    if k == "tag":
                        continue
                    set_dotted(vd, k, v)
                cfg = deep_merge(cfg, vd)
                tagparts.append(var.get("tag", "_".join(f"{short(k)}={fmt(v)}" for k, v in var.items())))
            tag = "__".join(tagparts) or "default"
            for seed in spec.get("seeds", [0]):
                c = deep_merge(cfg, {"seed": seed})
                runs.append((root / tag / f"seed{seed}", c))
    return runs


def run_one(run_dir: Path, cfg: dict, threads: int) -> tuple[Path, int, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "requested_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    t0 = time.time()
    with open(run_dir / "stdout.log", "w") as log:
        rc = subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
             "--run-dir", str(run_dir), "--threads", str(threads)],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
        )
    return run_dir, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per job")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run only every n-th job starting at i")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    runs = expand(spec, parse_overrides(args.set))
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        runs = runs[i::n]
    todo = [(d, c) for d, c in runs if not (d / "eval.json").exists()]
    steps = sum(int(c["ppo"]["total_timesteps"]) for _, c in todo)
    print(f"[{spec['name']}] {len(runs)} runs, {len(runs) - len(todo)} done, "
          f"{len(todo)} to go ({steps / 1e6:.1f}M env steps)")
    if args.dry_run:
        for d, _ in todo:
            print("  ", d)
        return

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one, d, c, args.threads) for d, c in todo]
        for k, f in enumerate(as_completed(futs), 1):
            d, rc, dt = f.result()
            status = "ok" if rc == 0 else f"FAILED (rc={rc}, see {d / 'stdout.log'})"
            print(f"  [{k}/{len(todo)}] {d}  {dt / 60:.1f} min  {status}", flush=True)
            if rc != 0:
                failed.append(d)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
