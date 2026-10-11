#!/usr/bin/env python
"""Figures and tables for the three extension studies.

    python scripts/analyze_extras.py                 # whatever inputs exist
    python scripts/analyze_extras.py --mode deterministic

Inputs
------
results/braid.json     from scripts/eval_braid.py
runs/memory_size/      from scripts/run_experiments.py configs/experiments/memory_size.yaml
results/probe.json     from scripts/probe_lstm.py

Outputs: paper/figures/{braid,memory_size,probe}.{pdf,png}, paper/tables/{braid,memory_size,probe}.tex
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import GRID, INK, MEM_STYLE, MUTED, _style, bootstrap_ci  # noqa: E402

PROBE_LABELS = {
    "heading": "heading", "last_action": "last action", "visited_up": "visited up",
    "visited_right": "visited right", "visited_down": "visited down", "visited_left": "visited left",
    "food_dir": "food direction*", "nest_dx": "nest dx", "nest_dy": "nest dy",
    "n_unique": "#cells visited", "food_dist": "food distance*",
}
SRC_STYLE = {
    "lstm_h": dict(color="#1baf7a", label="trained LSTM"),
    "random_h": dict(color="#9a99a0", label="untrained LSTM"),
    "obs": dict(color="#2a78d6", label="current observation"),
}


def ci_str(vals, pct=True):
    lo, hi = bootstrap_ci(vals)
    f = 100 if pct else 1
    return f"{np.mean(vals) * f:.1f} {{\\scriptsize[{lo * f:.1f}, {hi * f:.1f}]}}"


# ------------------------------------------------------------------- braid
def braid(res_dir: Path, fig_dir: Path, tab_dir: Path, mode: str, plt):
    path = res_dir / "braid.json"
    if not path.exists():
        print("[braid] results/braid.json missing (run scripts/eval_braid.py)")
        return
    data = json.loads(path.read_text())
    levels = [str(l) for l in data["levels"]]
    xs = [float(l) for l in levels]
    groups = defaultdict(list)
    for rd, r in data["runs"].items():
        mem = r["memory"] if r["memory"] != "stack" else "stack"
        trained = "braid" if r["train_braid"] > 0 else "perfect"
        groups[(mem, trained)].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, ylab in zip(axes, ("success_rate", "spl"), ("Success rate", "SPL")):
        for name, ls in (("wall_follower", "--"), ("random", ":"), ("oracle", "-.")):
            ys = [data["baselines"][name][l][metric] for l in levels]
            ax.plot(xs, ys, color=MUTED, ls=ls, lw=1.2, label=name.replace("_", " "))
        for (mem, trained), runs in sorted(groups.items()):
            st = MEM_STYLE.get(mem, dict(color=INK, marker="o", label=mem))
            ys, lo, hi = [], [], []
            for l in levels:
                vals = [r["levels"][l][mode][metric] for r in runs]
                ys.append(np.mean(vals))
                a, b = bootstrap_ci(vals)
                lo.append(a)
                hi.append(b)
            lab = f"{st['label']} (trained on {trained})"
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ms=4, lw=1.6,
                    ls="-" if trained == "perfect" else (0, (4, 1.5)), label=lab)
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.set_xlabel("Braid level (fraction of dead ends opened)", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "braid.pdf")
    fig.savefig(fig_dir / "braid.png", dpi=200)
    plt.close(fig)

    rows = [r"\begin{tabular}{l" + "c" * len(levels) + "}", r"\toprule",
            "Success (\\%), braid level & " + " & ".join(levels) + r" \\", r"\midrule"]
    for name in ("random", "wall_follower", "oracle"):
        rows.append(name.replace("_", " ") + " & "
                    + " & ".join(f"{100 * data['baselines'][name][l]['success_rate']:.1f}" for l in levels) + r" \\")
    for (mem, trained), runs in sorted(groups.items()):
        lab = f"{MEM_STYLE.get(mem, {}).get('label', mem)} ({trained})"
        rows.append(lab + " & " + " & ".join(
            ci_str([r["levels"][l][mode]["success_rate"] for r in runs]) for l in levels) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tab_dir / "braid.tex").write_text("\n".join(rows))
    print("[braid] wrote figure + table")


# ------------------------------------------------------------- memory size
def memory_size(runs_dir: Path, res_dir: Path, fig_dir: Path, tab_dir: Path, mode: str, plt):
    exp = runs_dir / "memory_size"
    by_h = defaultdict(list)
    for ev in sorted(exp.glob("*/seed*/eval.json")):
        cfg = yaml.safe_load((ev.parent / "config.yaml").read_text())
        h = json.loads(ev.read_text())["heldout"][mode]
        by_h[int(cfg["agent"]["lstm_hidden"])].append(h)
    if not by_h:
        print("[memory_size] no finished runs")
        return
    hs = sorted(by_h)
    bpath = res_dir / "baselines.json"
    base = json.loads(bpath.read_text()) if bpath.exists() else {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, metric, ylab in zip(axes, ("success_rate", "spl"), ("Held-out success rate", "Held-out SPL")):
        mean, lo, hi = [], [], []
        for h in hs:
            vals = [r[metric] for r in by_h[h]]
            ax.scatter([h] * len(vals), vals, s=9, color="#1baf7a", alpha=0.35, lw=0, zorder=2)
            mean.append(np.mean(vals))
            a, b = bootstrap_ci(vals)
            lo.append(a)
            hi.append(b)
        ax.plot(hs, mean, color="#1baf7a", marker="^", ms=4, lw=1.6, zorder=3, label="LSTM")
        ax.fill_between(hs, lo, hi, color="#1baf7a", alpha=0.15, lw=0)
        for name, ls in (("wall_follower", "--"), ("random", ":")):
            if name in base:
                ax.axhline(base[name]["11"][metric], color=MUTED, lw=1, ls=ls, label=name.replace("_", " "))
        ax.set_xscale("log", base=2)
        ax.set_xticks(hs)
        ax.set_xticklabels([str(h) for h in hs])
        ax.set_xlabel("LSTM hidden units", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "memory_size.pdf")
    fig.savefig(fig_dir / "memory_size.png", dpi=200)
    plt.close(fig)
    rows = [r"\begin{tabular}{rccc}", r"\toprule",
            r"Hidden units & Success (\%) & SPL (\%) & Seeds \\", r"\midrule"]
    for h in hs:
        rows.append(f"{h} & {ci_str([r['success_rate'] for r in by_h[h]])} & "
                    f"{ci_str([r['spl'] for r in by_h[h]])} & {len(by_h[h])} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tab_dir / "memory_size.tex").write_text("\n".join(rows))
    print("[memory_size] wrote figure + table")


# -------------------------------------------------------------------- probe
def probe(res_dir: Path, fig_dir: Path, tab_dir: Path, plt):
    path = res_dir / "probe.json"
    if not path.exists():
        print("[probe] results/probe.json missing (run scripts/probe_lstm.py)")
        return
    runs = json.loads(path.read_text())
    targets = list(PROBE_LABELS)
    score = {s: {t: [] for t in targets} for s in SRC_STYLE}
    for r in runs:
        for s in SRC_STYLE:
            for t in targets:
                v = r["probes"].get(s, {}).get(t)
                if v is None:
                    continue
                # classification: accuracy gain over majority class; regression: R^2
                score[s][t].append(v["acc"] - v["majority"] if "acc" in v else v["r2"])
    rhr = [r["rhr_agreement"] for r in runs]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.27
    for k, (s, st) in enumerate(SRC_STYLE.items()):
        vals = [np.mean(score[s][t]) if score[s][t] else np.nan for t in targets]
        err = []
        for t in targets:
            if len(score[s][t]) > 1:
                a, b = bootstrap_ci(score[s][t])
                err.append((np.mean(score[s][t]) - a, b - np.mean(score[s][t])))
            else:
                err.append((0, 0))
        ax.bar(np.arange(len(targets)) + (k - 1) * w, vals, w, color=st["color"], label=st["label"],
               yerr=np.array(err).T, error_kw=dict(lw=0.8, ecolor=MUTED))
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([PROBE_LABELS[t] for t in targets], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Probe score  (acc. gain over majority | $R^2$)", fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "probe.pdf")
    fig.savefig(fig_dir / "probe.png", dpi=200)
    plt.close(fig)

    rows = [r"\begin{tabular}{lccc}", r"\toprule",
            r"Target & Trained LSTM & Untrained LSTM & Observation \\", r"\midrule"]
    for t in targets:
        rows.append(PROBE_LABELS[t] + " & " + " & ".join(
            f"{np.mean(score[s][t]):.2f}" if score[s][t] else "--" for s in SRC_STYLE) + r" \\")
    rows += [r"\midrule", rf"Right-hand-rule agreement & \multicolumn{{3}}{{c}}{{{np.mean(rhr):.2f} (chance 0.25)}} \\",
             r"\bottomrule", r"\end{tabular}"]
    (tab_dir / "probe.tex").write_text("\n".join(rows))
    print(f"[probe] {len(runs)} runs; right-hand-rule agreement {np.mean(rhr):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--mode", default="stochastic", choices=["stochastic", "deterministic"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "axes.labelcolor": MUTED})

    fig_dir, tab_dir = Path(args.paper) / "figures", Path(args.paper) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)
    braid(res, fig_dir, tab_dir, args.mode, plt)
    memory_size(Path(args.runs), res, fig_dir, tab_dir, args.mode, plt)
    probe(res, fig_dir, tab_dir, plt)


if __name__ == "__main__":
    main()
