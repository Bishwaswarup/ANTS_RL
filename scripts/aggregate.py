#!/usr/bin/env python
"""Aggregate finished runs into CSVs, LaTeX tables and paper figures.

    python scripts/aggregate.py                 # every experiment under runs/
    python scripts/aggregate.py --exp main reward

Outputs
-------
results/<exp>/runs.csv        one row per (run, eval set, mode)
results/<exp>/summary.csv     mean and 95% bootstrap CI over seeds
paper/tables/<exp>.tex        booktabs table (held-out, stochastic + deterministic)
paper/figures/<exp>_*.pdf     figures used by the paper (if a recipe exists)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Categorical slots 1-3 of the validated reference palette (all-pairs safe),
# plus distinct markers so identity never relies on colour alone.
MEM_STYLE = {
    "none":  dict(color="#2a78d6", marker="o", label="No memory"),
    "stack": dict(color="#eb6834", marker="s", label="Frame stack (4)"),
    "lstm":  dict(color="#1baf7a", marker="^", label="LSTM"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["success_rate", "spl", "mean_steps", "wall_hit_rate", "mean_unique_cells"]


def bootstrap_ci(x, n=5000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def collect(exp_dir: Path) -> pd.DataFrame:
    rows = []
    for ev in sorted(exp_dir.glob("*/seed*/eval.json")):
        run = ev.parent
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        data = json.loads(ev.read_text())
        base = dict(
            tag=run.parent.name, seed=cfg["seed"], memory=cfg["agent"]["memory"],
            n_train_mazes=cfg["env"]["n_train_mazes"], reward=cfg["env"]["reward"]["preset"],
            obs_window=cfg["env"]["obs_window"], size=cfg["env"]["width"],
            timesteps=data.get("timesteps"), train_time_s=data.get("train_time_s"),
        )
        sets = {"heldout": data["heldout"], "train_dist": data["train_dist"]}
        for size, d in data.get("transfer", {}).items():
            sets[f"transfer_{size}"] = d
        for set_name, modes in sets.items():
            for mode, m in modes.items():
                rows.append({**base, "set": set_name, "mode": mode, **{k: m.get(k) for k in METRICS}})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["tag", "memory", "n_train_mazes", "reward", "obs_window", "size", "set", "mode"]
    out = []
    for k, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, k), n_seeds=len(g))
        for m in METRICS:
            row[m] = g[m].mean()
            row[f"{m}_lo"], row[f"{m}_hi"] = bootstrap_ci(g[m].values)
        out.append(row)
    return pd.DataFrame(out)


def pool_label(n):
    return r"$\infty$" if int(n) == 0 else str(int(n))


# ------------------------------------------------------------------ tables
def latex_table(summ: pd.DataFrame, exp: str, cols: list[str]) -> str:
    s = summ[summ["set"] == "heldout"]
    piv = {}
    for _, r in s.iterrows():
        key = tuple(r[c] for c in cols)
        piv.setdefault(key, {})[r["mode"]] = r
    def cell(r, m, pct=True):
        if r is None:
            return "--"
        f = 100 if pct else 1
        return f"{r[m] * f:.1f} {{\\scriptsize[{r[m + '_lo'] * f:.1f}, {r[m + '_hi'] * f:.1f}]}}"
    names = {"memory": "Memory", "n_train_mazes": "Train layouts", "reward": "Reward",
             "obs_window": "Window $k$"}
    head = " & ".join(names.get(c, c) for c in cols)
    lines = [
        r"\begin{tabular}{" + "l" * len(cols) + "cccc}", r"\toprule",
        head + r" & \multicolumn{2}{c}{Stochastic} & \multicolumn{2}{c}{Deterministic} \\",
        r"\cmidrule(lr){" + f"{len(cols)+1}-{len(cols)+2}" + r"}\cmidrule(lr){" + f"{len(cols)+3}-{len(cols)+4}" + "}",
        " & " * len(cols) + r"Success (\%) & SPL (\%) & Success (\%) & SPL (\%) \\", r"\midrule",
    ]
    def fmtkey(c, v):
        if c == "memory":
            return MEM_STYLE.get(v, {}).get("label", v)
        if c == "n_train_mazes":
            return pool_label(v)
        return str(v)
    for key in sorted(piv, key=lambda k: tuple(str(x) for x in k)):
        d = piv[key]
        st, de = d.get("stochastic"), d.get("deterministic")
        lines.append(" & ".join(fmtkey(c, v) for c, v in zip(cols, key)) + " & "
                     + " & ".join([cell(st, "success_rate"), cell(st, "spl"),
                                   cell(de, "success_rate"), cell(de, "spl")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def transfer_table(summ: pd.DataFrame, mode="stochastic") -> str:
    """Success (%) on held-out layouts of increasing size, same window."""
    s = summ[summ["mode"] == mode]
    sets = ["heldout"] + sorted({x for x in s["set"] if x.startswith("transfer_")},
                                key=lambda x: int(x.split("_")[1]))
    sizes = [None] + [int(x.split("_")[1]) for x in sets[1:]]
    keys = sorted({(r.memory, r.n_train_mazes) for r in s.itertuples()},
                  key=lambda k: (list(MEM_STYLE).index(k[0]) if k[0] in MEM_STYLE else 9, k[1] == 0, k[1]))
    base_size = int(summ["size"].iloc[0]) if "size" in summ else 11
    head = " & ".join(f"${(z or base_size)}\\times{(z or base_size)}$" for z in sizes)
    lines = [r"\begin{tabular}{ll" + "c" * len(sets) + "}", r"\toprule",
             r"Memory & Train layouts & " + head + r" \\", r"\midrule"]
    for mem, pool in keys:
        cells = []
        for st in sets:
            r = s[(s.memory == mem) & (s.n_train_mazes == pool) & (s["set"] == st)]
            cells.append("--" if r.empty else f"{100 * r.success_rate.iloc[0]:.1f}")
        lines.append(f"{MEM_STYLE.get(mem, {}).get('label', mem)} & {pool_label(pool)} & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def baselines_table(baselines: dict) -> str:
    """Success / SPL (%) of the non-learned baselines, straight from results/baselines.json."""
    names = {"random": "Random walk", "wall_follower": "Wall follower", "oracle": "Oracle"}
    sizes = sorted({int(z) for b in baselines.values() for z in b})
    lines = [r"\begin{tabular}{@{}l" + "cc" * len(sizes) + "@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{${z}\times{z}$}}" for z in sizes) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(sizes))),
             "Baseline & " + " & ".join(["Success & SPL"] * len(sizes)) + r" \\", r"\midrule"]
    for key in ("random", "wall_follower", "oracle"):
        if key not in baselines:
            continue
        cells = []
        for z in sizes:
            m = baselines[key].get(str(z))
            cells += (["--", "--"] if m is None else
                      [f"{100 * m['success_rate']:.1f}", f"{100 * m['spl']:.1f}"])
        lines.append(f"{names[key]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- figures
def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _baseline_lines(ax, baselines, size="11", metric="success_rate"):
    if not baselines:
        return
    for name, ls in (("wall_follower", "--"), ("random", ":")):
        if name in baselines and size in baselines[name]:
            v = baselines[name][size][metric]
            ax.axhline(v, color=MUTED, lw=1, ls=ls, zorder=1)
            ax.annotate(name.replace("_", " "), (1.0, v), xycoords=("axes fraction", "data"),
                        xytext=(-2, 3), textcoords="offset points", ha="right",
                        fontsize=7, color=MUTED)


def fig_main(summ, runs_dir: Path, out: Path, baselines, mode="stochastic"):
    import matplotlib.pyplot as plt
    pools = sorted(summ["n_train_mazes"].unique(), key=lambda n: (n == 0, n))
    xs = np.arange(len(pools))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, set_name, title in ((axes[0], "train_dist", "Training layouts"),
                                (axes[1], "heldout", "Held-out layouts")):
        for mem, st in MEM_STYLE.items():
            s = summ[(summ.memory == mem) & (summ.set == set_name) & (summ["mode"] == mode)]
            if s.empty:
                continue
            s = s.set_index("n_train_mazes").reindex(pools)
            y, lo, hi = s.success_rate.values, s.success_rate_lo.values, s.success_rate_hi.values
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
            ax.plot(xs, y, color=st["color"], marker=st["marker"], ms=5, lw=2, label=st["label"])
        if set_name == "heldout":
            _baseline_lines(ax, baselines)
        ax.set_xticks(xs, [pool_label(p).replace("$\\infty$", "∞") for p in pools])
        ax.set_xlabel("# distinct training layouts", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].set_ylabel(f"Success rate ({mode})", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "main_generalisation.pdf")
    fig.savefig(out / "main_generalisation.png", dpi=200)
    plt.close(fig)
    fig_curves(runs_dir, out / "main_curves", pool=0, mode=mode)


def fig_curves(runs_dir: Path, out_stem: Path, pool=0, mode="stochastic"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    any_curve = False
    for mem, st in MEM_STYLE.items():
        curves = []
        for run in runs_dir.glob("*/seed*"):
            cfgp, cp = run / "config.yaml", run / "curves.csv"
            if not (cfgp.exists() and cp.exists()):
                continue
            cfg = yaml.safe_load(cfgp.read_text())
            if cfg["agent"]["memory"] != mem or cfg["env"]["n_train_mazes"] != pool:
                continue
            curves.append(pd.read_csv(cp).set_index("timesteps")[f"{mode}_success"])
        if not curves:
            continue
        any_curve = True
        df = pd.concat(curves, axis=1).sort_index().interpolate(limit_area="inside")
        x = df.index.values / 1e6
        m = df.mean(axis=1).values
        sd = df.std(axis=1).fillna(0).values
        ax.fill_between(x, m - sd, m + sd, color=st["color"], alpha=0.15, lw=0)
        ax.plot(x, m, color=st["color"], lw=2, label=st["label"])
    if not any_curve:
        plt.close(fig)
        return
    ax.set_xlabel("Environment steps (M)", fontsize=8, color=MUTED)
    ax.set_ylabel(f"Held-out success ({mode})", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.02)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    plt.close(fig)


def fig_bars(summ, out_stem: Path, group: str, sub: str, baselines=None, mode="stochastic"):
    """Grouped bars: x = ``group`` values, bars = ``sub`` values; success and SPL panels."""
    import matplotlib.pyplot as plt
    s = summ[(summ.set == "heldout") & (summ["mode"] == mode)]
    groups = sorted(s[group].unique(), key=str)
    subs = sorted(s[sub].unique(), key=lambda v: (v == 0, v) if sub == "n_train_mazes" else str(v))
    palette = ["#2a78d6", "#eb6834", "#1baf7a"]
    hatches = ["", "//", ".."]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    w = 0.8 / max(len(subs), 1)
    for ax, metric, title in ((axes[0], "success_rate", "Success rate"), (axes[1], "spl", "SPL")):
        for j, sv in enumerate(subs):
            for i, gv in enumerate(groups):
                r = s[(s[group] == gv) & (s[sub] == sv)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i + (j - (len(subs) - 1) / 2) * w
                ax.bar(x, r[metric], w * 0.92, color=palette[j % 3], hatch=hatches[j % 3],
                       edgecolor="white", lw=0, label=pool_label(sv) if (sub == "n_train_mazes" and i == 0)
                       else (str(sv) if i == 0 else None))
                ax.errorbar(x, r[metric], yerr=[[r[metric] - r[metric + "_lo"]], [r[metric + "_hi"] - r[metric]]],
                            color=INK, lw=1, capsize=2)
        if metric == "success_rate":
            _baseline_lines(ax, baselines)
        labels = [MEM_STYLE[g]["label"] if group == "memory" else
                  (pool_label(g) if group == "n_train_mazes" else str(g)) for g in groups]
        ax.set_xticks(range(len(groups)), [l.replace("$\\infty$", "∞") for l in labels], fontsize=8)
        ax.set_title(title + f" ({mode}, held-out)", fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 1.05)
        _style(ax)
    sub_title = {"n_train_mazes": "train layouts", "memory": "memory", "obs_window": "window k",
                 "reward": "reward"}.get(sub, sub)
    leg_labels = None
    h, l = axes[0].get_legend_handles_labels()
    if sub == "n_train_mazes":
        leg_labels = [x.replace("$\\infty$", "∞") for x in l]
    axes[0].legend(h, leg_labels or l, title=sub_title, frameon=False, fontsize=7, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=200)
    plt.close(fig)


RECIPES = {
    "main": (["memory", "n_train_mazes"], lambda summ, rd, out, b: fig_main(summ, rd, out, b)),
    "reward": (["reward", "n_train_mazes"],
               lambda summ, rd, out, b: fig_bars(summ, out / "reward", "reward", "n_train_mazes", b)),
    "window": (["obs_window", "memory"],
               lambda summ, rd, out, b: fig_bars(summ, out / "window", "obs_window", "memory", b)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--exp", nargs="*", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "axes.labelcolor": MUTED})

    runs_root = Path(args.runs)
    exps = args.exp or (sorted(p.name for p in runs_root.iterdir() if p.is_dir())
                        if runs_root.exists() else [])
    bpath = Path(args.results) / "baselines.json"
    baselines = json.loads(bpath.read_text()) if bpath.exists() else None
    fig_dir, tab_dir = Path(args.paper) / "figures", Path(args.paper) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    if baselines:
        (tab_dir / "baselines.tex").write_text(baselines_table(baselines))
        print(f"[baselines] wrote {tab_dir / 'baselines.tex'}")

    for exp in exps:
        df = collect(runs_root / exp)
        if df.empty:
            print(f"[{exp}] no finished runs")
            continue
        out = Path(args.results) / exp
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "runs.csv", index=False)
        summ = summarise(df)
        summ.to_csv(out / "summary.csv", index=False)
        cols, recipe = RECIPES.get(exp, (["tag"], None))
        (tab_dir / f"{exp}.tex").write_text(latex_table(summ, exp, cols))
        if exp == "main" and any(summ["set"].str.startswith("transfer_")):
            (tab_dir / "main_transfer.tex").write_text(transfer_table(summ))
        if recipe:
            recipe(summ, runs_root / exp, fig_dir, baselines)
        h = summ[(summ.set == "heldout")]
        print(f"[{exp}] {df.seed.nunique()} seeds, {df.tag.nunique()} configs")
        print(h.pivot_table(index="tag", columns="mode", values="success_rate").round(3).to_string())


if __name__ == "__main__":
    main()
