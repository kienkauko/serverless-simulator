"""Plot PPO training curves for the two arrival traces (day_night, non_station).

Reads scalar logs from the SB3 TensorBoard event files in
  ../logs/ppo_idle_timeout_day_night_8/
  ../logs/ppo_idle_timeout_non_station_1/
and saves one PNG per metric in this directory.

Run: python -m dynamic_pool.RL.plot_train_results.plot_training
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tbparse import SummaryReader


# ----- style (individual plots) -----
LINE_WIDTH = 3.0
MARKER_SIZE = 9
LABEL_SIZE = 17
TICK_SIZE = 16
LEGEND_SIZE = 16
LEGEND_TITLE_SIZE = 16
FIGURE_SIZE = (7, 5)
CI_ALPHA = 0.2

# ----- style (combined 3x2 grid for single-column paper) -----
# Subplots are narrow, so bump fonts / lines / markers to stay readable.
COMBINED_FIGSIZE = (12, 6.5)          # whole figure (3 cols x 2 rows)
COMBINED_LINE_WIDTH = 3.5
COMBINED_MARKER_SIZE = 11
COMBINED_LABEL_SIZE = 20
COMBINED_TICK_SIZE = 17
COMBINED_LEGEND_SIZE = 20

plt.rcParams.update({
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_TITLE_SIZE,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
})


_HERE = Path(__file__).resolve().parent
_RL_DIR = _HERE.parent
_LOGS_DIR = _RL_DIR / "logs"

RUNS = {
    "day_night":   _LOGS_DIR / "ppo_idle_timeout_day_night_8",
    "non_station": _LOGS_DIR / "ppo_idle_timeout_non_station_1",
}

# Legend labels for each run
LABELS = {
    "day_night":   "D-N",
    "non_station": "N-S",
}

_CMAP = plt.cm.tab10
COLORS = {
    "day_night":   _CMAP(0),
    "non_station": _CMAP(1),
}

MARKERS = {
    "day_night":   "o",
    "non_station": "s",
}

# (tag, ylabel, smoothing_window, output_filename)
METRICS = [
    ("rollout/ep_rew_mean", "Episode reward",   3, "episode_reward.png"),
    ("sim/cold_ratio",      "Cold-start ratio", 5, "cold_start_ratio.png"),
    ("sim/ram_util",        "RAM utilization",  5, "ram_utilization.png"),
    ("sim/idle_timeout",    "Idle timeout (s)", 5, "idle_timeout.png"),
    ("train/value_loss",    "Value loss",       3, "value_loss.png"),
    ("train/entropy_loss",  "Entropy loss",     3, "entropy_loss.png"),
]


def load_scalars(log_dir: Path) -> pd.DataFrame:
    if not log_dir.exists():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")
    reader = SummaryReader(str(log_dir))
    return reader.scalars


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average for visual clarity. Falls back to raw if too short."""
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def get_series(df: pd.DataFrame, tag: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["tag"] == tag].sort_values("step")
    return sub["step"].to_numpy(), sub["value"].to_numpy()


def plot_metric(
    runs_data: dict[str, pd.DataFrame],
    tag: str,
    ylabel: str,
    out_path: Path,
    smooth_window: int = 5,
    yscale: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for key, df in runs_data.items():
        steps, vals = get_series(df, tag)
        if len(steps) == 0:
            print(f"  [warn] tag {tag!r} not in run {key!r}; skipping")
            continue
        steps_k = steps / 1000.0  # steps -> k-steps for readability
        smoothed = smooth(vals, window=smooth_window)
        ax.plot(
            steps_k, smoothed,
            color=COLORS[key],
            marker=MARKERS[key],
            markevery=max(1, len(steps_k) // 12),
            label=LABELS[key],
        )

    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    # fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    # print(f"  saved {out_path.name}")


def combine(
    runs_data: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Plot all six metrics on one figure (3 cols x 2 rows), single shared
    legend at the top. Sized for a single-column paper: narrow subplots,
    enlarged fonts / linewidth / markersize so they remain readable when
    the figure is scaled to fit the column."""
    fig, axes = plt.subplots(2, 3, figsize=COMBINED_FIGSIZE)
    axes_flat = axes.flat

    shared_handles = None
    shared_labels = None

    for ax, (tag, ylabel, smooth_w, _fname) in zip(axes_flat, METRICS):
        for key, df in runs_data.items():
            steps, vals = get_series(df, tag)
            if len(steps) == 0:
                print(f"  [warn] tag {tag!r} not in run {key!r}; skipping")
                continue
            steps_k = steps / 1000.0
            smoothed = smooth(vals, window=smooth_w)
            ax.plot(
                steps_k, smoothed,
                color=COLORS[key],
                marker=MARKERS[key],
                markevery=max(1, len(steps_k) // 10),
                linewidth=COMBINED_LINE_WIDTH,
                markersize=COMBINED_MARKER_SIZE,
                label=LABELS[key],
            )

        ax.set_ylabel(ylabel, fontsize=COMBINED_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=COMBINED_TICK_SIZE)
        ax.grid(True, alpha=0.3)

        if shared_handles is None:
            shared_handles, shared_labels = ax.get_legend_handles_labels()

    # xlabel only on the bottom row (top row shares the same x-axis meaning).
    for ax in axes[1, :]:
        ax.set_xlabel("Steps (k)", fontsize=COMBINED_LABEL_SIZE)

    # Single shared legend across the top.
    fig.legend(
        shared_handles, shared_labels,
        loc="upper center",
        ncol=len(shared_labels) if shared_labels else 1,
        fontsize=COMBINED_LEGEND_SIZE,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    # fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    # print(f"  saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combine", action="store_true",
        help="Plot all 6 metrics in one figure (3x2 grid) for single-column paper",
    )
    args = parser.parse_args()

    runs_data = {key: load_scalars(p) for key, p in RUNS.items()}
    for key, df in runs_data.items():
        print(f"{key:12s} -> {len(df)} scalar rows, "
              f"{df['tag'].nunique()} tags")

    out = _HERE

    print("\nPlotting...")

    if args.combine:
        combine(runs_data, out_path=out / "combined.png")
    else:
        for tag, ylabel, smooth_w, fname in METRICS:
            plot_metric(
                runs_data,
                tag=tag,
                ylabel=ylabel,
                out_path=out / fname,
                smooth_window=smooth_w,
            )

    print(f"\nAll plots written to {out}")


if __name__ == "__main__":
    main()
