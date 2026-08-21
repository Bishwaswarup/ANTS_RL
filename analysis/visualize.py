"""
analysis/visualize.py
=====================
Publication-quality plotting tools for EAC experiments.

All functions return matplotlib Figure objects so they can be shown
interactively or saved to disk::

    fig = plot_visit_heatmap(heatmap, maze_grid)
    fig.savefig("analysis/plots/heatmap.png", dpi=150, bbox_inches="tight")

Functions
---------
  plot_visit_heatmap      — agent visitation frequency overlaid on the maze
  plot_training_curves    — reward / episode-length curves from TensorBoard logs
  plot_success_rate       — rolling success rate over training
  plot_path_comparison    — side-by-side maze with optimal vs learned path
  plot_episode_summary    — multi-panel summary of a single evaluation run
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

WALL_COLOR     = "#2C2C2C"
PATH_COLOR     = "#F5F5F0"
NEST_COLOR     = "#8B4513"   # saddle brown
FOOD_COLOR     = "#228B22"   # forest green
ANT_COLOR      = "#DC143C"   # crimson
HEAT_CMAP      = LinearSegmentedColormap.from_list(
    "ant_heat", ["#F5F5F0", "#FFD700", "#FF6347", "#8B0000"]
)
REWARD_COLOR   = "#4169E1"   # royal blue
LENGTH_COLOR   = "#2E8B57"   # sea green
SUCCESS_COLOR  = "#FF8C00"   # dark orange

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ---------------------------------------------------------------------------
# 1. Visitation Heatmap
# ---------------------------------------------------------------------------

def plot_visit_heatmap(
    heatmap: np.ndarray,
    maze_grid: np.ndarray,
    nest_pos: Optional[tuple] = None,
    food_pos: Optional[tuple] = None,
    title: str = "Agent Visitation Heatmap",
    figsize: tuple = (7, 7),
) -> Figure:
    """
    Overlay agent visit frequency on the maze structure.

    Parameters
    ----------
    heatmap   : (W, H) int array — raw visit counts from MetricsCollector.
    maze_grid : (W, H) int array — 1=wall, 0=free.
    nest_pos  : (x, y) optional — drawn as a brown square.
    food_pos  : (x, y) optional — drawn as a green square.
    """
    W, H = maze_grid.shape
    display = np.zeros((H, W, 4))   # RGBA

    # Walls — dark fill
    wall_rgba = matplotlib.colors.to_rgba(WALL_COLOR)
    path_rgba = matplotlib.colors.to_rgba(PATH_COLOR)

    norm_heat = heatmap / (heatmap.max() + 1e-8)

    for x in range(W):
        for y in range(H):
            if maze_grid[x, y] == 1:
                display[y, x] = wall_rgba
            else:
                display[y, x] = HEAT_CMAP(norm_heat[x, y])

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display, origin="upper", interpolation="nearest")

    # Overlay nest / food markers
    if nest_pos:
        nx, ny = nest_pos
        ax.add_patch(mpatches.Rectangle(
            (nx - 0.5, ny - 0.5), 1, 1,
            linewidth=2, edgecolor=NEST_COLOR, facecolor=NEST_COLOR, alpha=0.85
        ))
        ax.text(nx, ny, "N", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")

    if food_pos:
        fx, fy = food_pos
        ax.add_patch(mpatches.Rectangle(
            (fx - 0.5, fy - 0.5), 1, 1,
            linewidth=2, edgecolor=FOOD_COLOR, facecolor=FOOD_COLOR, alpha=0.85
        ))
        ax.text(fx, fy, "F", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")

    # Colourbar
    sm = plt.cm.ScalarMappable(
        cmap=HEAT_CMAP, norm=plt.Normalize(0, heatmap.max())
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Total visits", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    return fig


# ---------------------------------------------------------------------------
# 2. Training Curves (from TensorBoard CSV or in-memory arrays)
# ---------------------------------------------------------------------------

def plot_training_curves(
    rewards: Sequence[float],
    lengths: Optional[Sequence[float]] = None,
    smoothing: int = 20,
    title: str = "Training Progress",
    figsize: tuple = (10, 4),
) -> Figure:
    """
    Plot mean episode reward (and optionally episode length) over training.

    Parameters
    ----------
    rewards   : sequence of per-episode rewards.
    lengths   : sequence of per-episode lengths (plotted on a secondary y-axis).
    smoothing : rolling-window size for the smoothed trend line.
    """
    rewards = np.array(rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)

    n_axes = 2 if lengths is not None else 1
    fig, axes = plt.subplots(1, n_axes, figsize=figsize)
    if n_axes == 1:
        axes = [axes]

    def _smooth(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")

    # Reward panel
    ax = axes[0]
    ax.plot(episodes, rewards, color=REWARD_COLOR, alpha=0.25, linewidth=0.8)
    smoothed = _smooth(rewards, smoothing)
    ax.plot(
        np.arange(smoothing, len(rewards) + 1), smoothed,
        color=REWARD_COLOR, linewidth=2, label=f"Reward (smooth {smoothing})"
    )
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Reward", fontsize=11)
    ax.legend(fontsize=9)

    # Length panel
    if lengths is not None:
        lengths = np.array(lengths, dtype=float)
        ax2 = axes[1]
        ax2.plot(episodes, lengths, color=LENGTH_COLOR, alpha=0.25, linewidth=0.8)
        smoothed_l = _smooth(lengths, smoothing)
        ax2.plot(
            np.arange(smoothing, len(lengths) + 1), smoothed_l,
            color=LENGTH_COLOR, linewidth=2, label=f"Length (smooth {smoothing})"
        )
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.set_title("Episode Length", fontsize=11)
        ax2.legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Rolling Success Rate
# ---------------------------------------------------------------------------

def plot_success_rate(
    successes: Sequence[bool],
    window: int = 50,
    title: str = "Rolling Success Rate",
    figsize: tuple = (8, 4),
) -> Figure:
    """
    Plot the rolling mean success rate.

    Parameters
    ----------
    successes : bool sequence — True if the episode was solved.
    window    : rolling window size.
    """
    s = np.array(successes, dtype=float)
    episodes = np.arange(1, len(s) + 1)

    kernel = np.ones(window) / window
    rolling = np.convolve(s, kernel, mode="valid")
    x_rolling = np.arange(window, len(s) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(episodes, s, alpha=0.15, color=SUCCESS_COLOR)
    ax.plot(x_rolling, rolling, color=SUCCESS_COLOR, linewidth=2.5,
            label=f"Rolling {window}-ep average")
    ax.axhline(0.75, color="grey", linestyle="--", linewidth=1, alpha=0.7,
               label="75 % threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Maze + path overlay
# ---------------------------------------------------------------------------

def plot_maze_path(
    maze_grid: np.ndarray,
    path: list[tuple],
    optimal_path: Optional[list[tuple]] = None,
    nest_pos: Optional[tuple] = None,
    food_pos: Optional[tuple] = None,
    title: str = "Agent Path",
    figsize: tuple = (6, 6),
) -> Figure:
    """
    Draw the maze and overlay the agent's trajectory.

    Parameters
    ----------
    maze_grid    : (W, H) int array.
    path         : list of (x, y) — agent trajectory.
    optimal_path : list of (x, y) — BFS shortest path (optional comparison).
    """
    W, H = maze_grid.shape
    display = np.zeros((H, W, 3))

    for x in range(W):
        for y in range(H):
            if maze_grid[x, y] == 1:
                display[y, x] = matplotlib.colors.to_rgb(WALL_COLOR)
            else:
                display[y, x] = matplotlib.colors.to_rgb(PATH_COLOR)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display, origin="upper", interpolation="nearest")

    def _draw_path(pts, color, label, lw=2.5, marker="o", ms=4):
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=lw, marker=marker,
                markersize=ms, label=label, alpha=0.85)

    if optimal_path:
        _draw_path(optimal_path, "#20B2AA", "Optimal (BFS)", lw=1.8, marker=None)

    _draw_path(path, ANT_COLOR, "Agent", lw=2.5)

    if nest_pos:
        ax.plot(*nest_pos, marker="s", color=NEST_COLOR, markersize=12,
                zorder=5, label="Nest")
    if food_pos:
        ax.plot(*food_pos, marker="*", color=FOOD_COLOR, markersize=14,
                zorder=5, label="Food")

    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Evaluation summary panel
# ---------------------------------------------------------------------------

def plot_episode_summary(
    metrics_summary: dict,
    figsize: tuple = (8, 4),
) -> Figure:
    """
    Render a bar chart summarising key evaluation metrics.

    Parameters
    ----------
    metrics_summary : dict — output of MetricsCollector.summary().
    """
    keys = ["success_rate", "mean_path_efficiency", "mean_wall_hit_rate"]
    labels = ["Success Rate", "Path Efficiency", "Wall Hit Rate"]
    colors = [SUCCESS_COLOR, REWARD_COLOR, "#CC4444"]
    values = [metrics_summary.get(k, 0.0) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: bar chart of rates
    ax = axes[0]
    bars = ax.barh(labels, values, color=colors, height=0.45)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Rate (0–1)")
    ax.set_title("Performance Metrics", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.1%}", va="center", fontsize=9)

    # Right: scalar stats table
    ax2 = axes[1]
    ax2.axis("off")
    stat_keys = ["n_episodes", "mean_steps", "mean_reward", "mean_unique_cells"]
    stat_labels = ["Episodes", "Mean Steps", "Mean Reward", "Mean Unique Cells"]
    table_data = [[lbl, f"{metrics_summary.get(k, '–'):.2f}"
                   if isinstance(metrics_summary.get(k), float) else str(metrics_summary.get(k, '–'))]
                  for lbl, k in zip(stat_labels, stat_keys)]
    table = ax2.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    ax2.set_title("Summary Statistics", fontsize=11, fontweight="bold", pad=20)

    fig.suptitle(
        f"Evaluation: {metrics_summary.get('run_name', 'EAC')}",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: save a figure
# ---------------------------------------------------------------------------

def save_fig(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    """Save `fig` to `path`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {path}")
