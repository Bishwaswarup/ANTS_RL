#!/usr/bin/env python
"""How diverse are the generated layouts, and do training and held-out layouts overlap?

Samples procedural training layouts exactly as the env does *before* the
train/test split is applied, and reports
  * how many of the samples are distinct layouts;
  * how many samples coincide with one of the held-out layouts, i.e. the
    overlap a seed-range-only split would have had;
  * the same count after the hash-based split (must be 0).

    python scripts/layout_stats.py --samples 200000 --heldout 500
Writes results/layout_stats.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ants_rl.env import EVAL_SEED_BASE, PROCEDURAL_SEED_BASE  # noqa: E402
from ants_rl.evaluation import heldout_episodes  # noqa: E402
from ants_rl.maze import layout_split, maze_from_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--out", default="results/layout_stats.json")
    args = ap.parse_args()
    z = args.size

    # Held-out layouts under the old scheme (consecutive eval seeds) and the new one (test split).
    old_held = {maze_from_seed(z, z, EVAL_SEED_BASE + i).tobytes() for i in range(args.heldout)}
    new_held = {maze_from_seed(z, z, s).tobytes()
                for s, _ in heldout_episodes(args.heldout, width=z, height=z)}

    rng = np.random.default_rng(0)
    seen, hits_old, hits_new, n_train_split = set(), 0, 0, 0
    for _ in range(args.samples):
        g = maze_from_seed(z, z, int(rng.integers(PROCEDURAL_SEED_BASE, 2**31 - 1)))
        b = g.tobytes()
        seen.add(b)
        hits_old += b in old_held
        if layout_split(g) == "train":  # what the env actually trains on
            n_train_split += 1
            hits_new += b in new_held
    out = {
        "size": z, "samples": args.samples, "distinct": len(seen),
        "heldout": args.heldout, "heldout_distinct_new": len(new_held),
        "hits_seed_range_split": hits_old,
        "train_split_samples": n_train_split, "hits_hash_split": hits_new,
    }
    print(json.dumps(out, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
