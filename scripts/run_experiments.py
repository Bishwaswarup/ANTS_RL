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

    # one run at a time, each maze in its own process, session capped at 3 h.
    # Re-run the same command to continue: finished runs are skipped and an
    # interrupted run resumes from its checkpoint.
    python scripts/run_experiments.py configs/experiments/main.yaml --subproc --max-hours 3
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


PAUSED = 3        # train.py exit code when it stops at the time budget
MIN_START_MIN = 5  # do not start a run with less than this many minutes left


def run_one(run_dir: Path, cfg: dict, threads: int, deadline: float | None):
    if deadline is not None and deadline - time.time() < 60 * MIN_START_MIN:
        return run_dir, None, 0.0  # skipped: session budget used up
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "requested_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
           "--run-dir", str(run_dir), "--threads", str(threads)]
    if deadline is not None:
        cmd += ["--max-hours", f"{(deadline - time.time()) / 3600:.4f}"]
    t0 = time.time()
    with open(run_dir / "stdout.log", "a") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return run_dir, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch threads per job")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: run only every n-th job starting at i")
    ap.add_argument("--subproc", action="store_true",
                    help="each maze in its own process (good with --jobs 1)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="session budget: running runs checkpoint and stop when it is used up, "
                         "no new run starts; re-run the command to continue")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.spec))
    extra = parse_overrides(args.set + (["ppo.vec_env=subproc"] if args.subproc else []))
    runs = expand(spec, extra)
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

    deadline = time.time() + 3600 * args.max_hours if args.max_hours else None
    failed, paused, skipped, done = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one, d, c, args.threads, deadline) for d, c in todo]
        for f in as_completed(futs):
            d, rc, dt = f.result()
            if rc is None:
                skipped += 1
                continue
            if rc == 0:
                done += 1
                status = "ok"
            elif rc == PAUSED:
                paused += 1
                status = "paused (time budget), will resume next session"
            else:
                failed.append(d)
                status = f"FAILED (rc={rc}, see {d / 'stdout.log'})"
            print(f"  {d}  {dt / 60:.1f} min  {status}", flush=True)
    left = len(todo) - done - len(failed)
    print(f"[{spec['name']}] this session: {done} finished, {paused} paused, "
          f"{len(failed)} failed; {left} still to do")
    if left:
        print("  re-run the same command to continue")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
