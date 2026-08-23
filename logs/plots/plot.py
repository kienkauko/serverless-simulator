import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── User-configurable style parameters ──
LINE_WIDTH   = 3.0
MARKER_SIZE  = 9
LABEL_SIZE   = 30
TICK_SIZE    = 30
LEGEND_SIZE  = 28
LEGEND_TITLE_SIZE = 28
BAR_SIZE     = 25     # text labels drawn on/near bars
CI_ALPHA     = 0.2
CAP_SIZE     = 16     # error-bar cap size
FIG_WIDTH    = 10
FIG_HEIGHT   = 7.5

# Style overrides for plot_time_series()
TS_LINE_WIDTH        = 2.5
TS_LABEL_SIZE        = 18
TS_TICK_SIZE         = 18
TS_LEGEND_SIZE       = 18
TS_LEGEND_TITLE_SIZE = 18
TS_TITLE_SIZE        = 18

# Apply global plotting styles
plt.rc('lines', linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
plt.rc('axes', labelsize=LABEL_SIZE, titlesize=LABEL_SIZE)
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('legend', fontsize=LEGEND_SIZE, title_fontsize=LEGEND_TITLE_SIZE)

TRAFFIC = "combined"  # Set to "day_night", "non_station", "combined", or None

# Per-traffic log directories
DIRS = {
    "day_night":   os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "day_night")),
    "non_station": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "non_station")),
}
TRAFFIC_ORDER  = ["day_night", "non_station"]
TRAFFIC_LABELS = {"day_night": "Bursted Diurnal", "non_station": "Non-Sta. Diurnal"}

COMBINED = TRAFFIC == "combined"

if TRAFFIC == "day_night":
    LOGS_DIR = DIRS["day_night"]
elif TRAFFIC == "non_station":
    LOGS_DIR = DIRS["non_station"]
elif TRAFFIC == "combined":
    LOGS_DIR = None
else:
    LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Skip the first 500 minutes — ML strategies only start applying at minute 501.
START_MINUTE = 501

# Filename format: YYYY_MM_DD_HHMMSS_multi_xN_<type>.csv
# <type> may itself contain underscores (e.g. "0_warm", "100_warm").
FNAME_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{6}_multi_x\d+_(?P<type>.+)\.csv$")

TYPE_ORDER = ["static", "SR", "analysis", "RL"]

TYPE_LABELS = {
    "static": "Baseline",
    "SR": "RES-Reserved",
    "analysis":  "RES-Dynamic",
    "RL":  "RL",
}


def discover_files(logs_dir):
    """Return dict: type -> list of file paths."""
    groups = {t: [] for t in TYPE_ORDER}
    for path in glob.glob(os.path.join(logs_dir, "*.csv")):
        name = os.path.basename(path)
        m = FNAME_RE.match(name)
        if not m:
            continue
        t = m.group("type")
        if t in groups:
            groups[t].append(path)
    return groups


def ci95(values):
    """Return 95% CI half-width for the mean of `values`."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return 0.0
    return 1.96 * np.std(v, ddof=1) / np.sqrt(n)


def load_repetitions(filepath):
    """Return a list of DataFrames, one per repetition.

    Splits on the 'repetition' column when present; otherwise detects
    repetition boundaries by a drop in the 'minute' counter.
    """
    df = pd.read_csv(filepath)
    if "repetition" in df.columns:
        return [rep.reset_index(drop=True)
                for _, rep in df.groupby("repetition", sort=True)]
    # Fallback: boundary = row where minute decreases
    boundaries = [0] + list(df.index[df["minute"].diff() < 0] + 1) + [len(df)]
    return [df.iloc[boundaries[i]:boundaries[i + 1]].reset_index(drop=True)
            for i in range(len(boundaries) - 1)]


def collect_per_minute(groups, col_func):
    """Pool a per-minute series across all repetitions for each type.

    col_func(df) -> 1D array of per-minute values.
    Returns dict: type -> pooled array.
    """
    out = {}
    for t, files in groups.items():
        pooled = []
        for f in files:
            for rep_df in load_repetitions(f):
                df = rep_df[rep_df["minute"] >= START_MINUTE]
                pooled.append(np.asarray(col_func(df), dtype=float))
        out[t] = np.concatenate(pooled) if pooled else np.array([])
    return out


def collect_per_run(groups, run_func):
    """One scalar per repetition. run_func(df) -> float."""
    out = {}
    for t, files in groups.items():
        vals = []
        for f in files:
            for rep_df in load_repetitions(f):
                vals.append(float(run_func(rep_df)))
        out[t] = np.asarray(vals, dtype=float)
    return out


def collect(groups, func, per_run=False):
    """Dispatch collection for single- or combined-traffic mode.

    Returns {type: array} normally, or {traffic: {type: array}} when COMBINED.
    """
    collector = collect_per_run if per_run else collect_per_minute
    if COMBINED:
        return {tr: collector(groups[tr], func) for tr in TRAFFIC_ORDER}
    return collector(groups, func)


def _window_rate(df, col):
    """Average rate of a cumulative column over [START_MINUTE, end].

    Subtracts the baseline at the row just before START_MINUTE so the result
    is not contaminated by the warm-up period.
    """
    window = df[df["minute"] >= START_MINUTE]
    if window.empty:
        return np.nan
    baseline = df[df["minute"] == START_MINUTE - 1]
    base_val = float(baseline[col].iloc[0]) if not baseline.empty else 0.0
    base_time = float(baseline["time"].iloc[0]) if not baseline.empty else 0.0
    last = window.iloc[-1]
    dt = float(last["time"]) - base_time
    return (float(last[col]) - base_val) / dt if dt > 0 else np.nan


def _means_cis(values_by_type):
    """Reduce a {type: array} dict to ordered (means, cis) lists."""
    means, cis = [], []
    for t in TYPE_ORDER:
        v = values_by_type.get(t, np.array([]))
        v = v[~np.isnan(v)] if v.size else v
        means.append(np.mean(v) if v.size else 0.0)
        cis.append(ci95(v))
    return means, cis


def bar_with_ci(values_by_type, ylabel, title, decimals=1, nudge_factor=0.2, ylim=None,
                legend_loc=None):
    tab10_colors = plt.get_cmap("tab10").colors
    labels = [TYPE_LABELS[t] for t in TYPE_ORDER]
    x = np.arange(len(labels))

    if COMBINED:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax.set_axisbelow(True)  # draw grid under the bars

        # day_night -> tab10[0] (no hatch), non_station -> tab10[2] (hatched)
        traffic_styles = {
            "day_night":   dict(color=tab10_colors[0], hatch=None),
            "non_station": dict(color=tab10_colors[2], hatch="///"),
        }
        n = len(TRAFFIC_ORDER)
        width = 0.8 / n

        # Nudge labels apart so adjacent bars' annotations don't overlap.
        # Left bar (j=0): shift left; right bar (j=n-1): shift right.
        label_nudge = width * nudge_factor

        for j, tr in enumerate(TRAFFIC_ORDER):
            means, cis = _means_cis(values_by_type[tr])
            offset = (j - (n - 1) / 2) * width
            style = traffic_styles[tr]
            ax.bar(x + offset, means, width, yerr=cis, ecolor='red',
                   capsize=CAP_SIZE, alpha=0.9, color=style["color"],
                   hatch=style["hatch"], label=TRAFFIC_LABELS[tr])

            sign = -1 if j < (n - 1) / 2 else (1 if j > (n - 1) / 2 else 0)
            text_dx = sign * label_nudge

            for i, (mean, ci) in enumerate(zip(means, cis)):
                if mean > 0 or ci > 0:
                    ax.text(x[i] + offset + text_dx, mean + ci,
                            f"{mean:.{decimals}f}",
                            ha='center', va='bottom', fontsize=BAR_SIZE,
                            bbox=dict(facecolor='white', alpha=0.1,
                                      edgecolor='none', pad=1))
        ax.legend(loc=legend_loc) if legend_loc else ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax.set_axisbelow(True)  # draw grid under the bars

        means, cis = _means_cis(values_by_type)
        ax.bar(x, means, yerr=cis, ecolor='red', capsize=CAP_SIZE, alpha=0.9,
               color=tab10_colors[:len(labels)])

        # Add text labels on top of each bar
        for i, (mean, ci) in enumerate(zip(means, cis)):
            if mean > 0 or ci > 0:
                ax.text(x[i], mean + ci, f"{mean:.{decimals}f}", ha='center',
                        va='bottom', fontsize=BAR_SIZE,
                        bbox=dict(facecolor='white', alpha=0.1,
                                  edgecolor='none', pad=1))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.1)

    ax.grid(color='lightgray', linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout()
    plt.show()


def plot_blocking(groups):
    def per_min(df):
            arr = df["arrival"].to_numpy(dtype=float)
            acc = df["accepted"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                br = np.where(arr > 0, 1.0 - acc / arr, np.nan)
            return br
    data = collect(groups, per_min)
    bar_with_ci(data, "Blocking rate", "Blocking rate by strategy (mean over minutes, 95% CI)")


def plot_power(groups):
    data = collect(groups, lambda df: _window_rate(df, "energy"), per_run=True)
    bar_with_ci(data, "Average Power (W)", "Power by strategy (95% CI across repetitions)",  decimals=1)


def plot_ram(groups):
    data = collect(groups, lambda df: _window_rate(df, "ram_area"), per_run=True)
    bar_with_ci(data, "Average RAMtime", "RAM by strategy (95% CI across repetitions)")


def _window_ram_per_request(df):
    """Total ram_area accumulated over [START_MINUTE, end] divided by total accepted requests."""
    window = df[df["minute"] >= START_MINUTE]
    if window.empty:
        return np.nan
    baseline = df[df["minute"] == START_MINUTE - 1]
    base_ram = float(baseline["ram_area"].iloc[0]) if not baseline.empty else 0.0
    last = window.iloc[-1]
    delta_ram = float(last["ram_area"]) - base_ram
    total_acc = float(window["accepted"].sum())
    return delta_ram / total_acc if total_acc > 0 else np.nan


def plot_ram_second_per_request(groups):
    data = collect(groups, _window_ram_per_request, per_run=True)
    bar_with_ci(data, "Mean RAM per Request (%.s)", "RAMsecond per request by strategy (95% CI across repetitions)")


def plot_cpu_second_per_request(groups):
    def per_run(df):
        window = df[df["minute"] >= START_MINUTE]
        if window.empty:
            return np.nan
        baseline = df[df["minute"] == START_MINUTE - 1]
        base_cpu = float(baseline["cpu_area"].iloc[0]) if not baseline.empty else 0.0
        last = window.iloc[-1]
        delta_cpu = float(last["cpu_area"]) - base_cpu
        total_acc = float(window["accepted"].sum())
        return delta_cpu / total_acc if total_acc > 0 else np.nan
    data = collect(groups, per_run, per_run=True)
    bar_with_ci(data, "Mean CPU per Request (%.s)", "CPUsecond per request by strategy (95% CI across repetitions)")


def plot_energy_per_request(groups):
    def per_run(df):
        window = df[df["minute"] >= START_MINUTE]
        if window.empty:
            return np.nan
        baseline = df[df["minute"] == START_MINUTE - 1]
        base_energy = float(baseline["energy"].iloc[0]) if not baseline.empty else 0.0
        last = window.iloc[-1]
        delta_energy = float(last["energy"]) - base_energy
        total_acc = float(window["accepted"].sum())
        return delta_energy / total_acc if total_acc > 0 else np.nan
    data = collect(groups, per_run, per_run=True)
    bar_with_ci(data, "Mean Energy per Request (J)", "Energy per request by strategy (95% CI across repetitions)")


def plot_latency(groups):
    data = collect(groups, lambda df: df["mean_latency"].to_numpy(dtype=float))
    bar_with_ci(data, "Mean processing time (s)", "Latency by strategy (mean over minutes, 95% CI)",
                decimals=2, nudge_factor=0.1, ylim=(1.0, 3.0))


def plot_ram_per_request(groups):
    def per_min(df):
        delta_ram = df["ram_area"].diff().to_numpy(dtype=float)
        dt = df["time"].diff().to_numpy(dtype=float)
        arr = df["arrival"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_ram = np.where(dt > 0, delta_ram / dt, np.nan)
            return np.where(arr > 0, avg_ram / arr, np.nan)
    data = collect(groups, per_min)
    bar_with_ci(data, "RAM per request", "RAM per request by strategy (95% CI across minutes)")


def plot_energy_per_request(groups):
    def per_min(df):
        delta_e = df["energy"].diff().to_numpy(dtype=float)
        arr = df["arrival"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(arr > 0, delta_e / arr, np.nan)
    data = collect(groups, per_min)
    bar_with_ci(data, "Energy per request (J)", "Energy per request by strategy (95% CI across minutes)", decimals=1)


def plot_cold_hit_percentage(groups):
    """Percentage of cold-hits over total requests arrived in the window.

    cold_hit is a cumulative column, so the window value is the delta between
    the last row and the baseline just before START_MINUTE. Requests arrived
    is a per-minute column, so the total is the sum over the window.
    """
    def per_run(df):
        window = df[df["minute"] >= START_MINUTE]
        if window.empty:
            return np.nan
        baseline = df[df["minute"] == START_MINUTE - 1]
        base_cold = float(baseline["cold_hit"].iloc[0]) if not baseline.empty else 0.0
        last = window.iloc[-1]
        delta_cold = float(last["cold_hit"]) - base_cold
        total_arr = float(window["accepted"].sum())
        return 100.0 * delta_cold / total_arr if total_arr > 0 else np.nan
    data = collect(groups, per_run, per_run=True)
    bar_with_ci(data, "Cold-hit rate (%)",
                "Cold-hit percentage by strategy (95% CI across repetitions)",
                decimals=2, legend_loc="center right")


def plot_cold_redundant(groups):
    tab10_colors = plt.get_cmap("tab10").colors
    labels = [TYPE_LABELS[t] for t in TYPE_ORDER]
    x = np.arange(len(labels))

    def _cold_red(g):
        """Return (cold_means, cold_cis, red_means, red_cis) for a plain groups dict."""
        cold = collect_per_minute(g, lambda df: df["cold_hit"].to_numpy(dtype=float))
        red  = collect_per_minute(g, lambda df: df["redundant"].to_numpy(dtype=float))
        cm = [np.mean(cold[t]) if cold[t].size else 0.0 for t in TYPE_ORDER]
        cc = [ci95(cold[t]) for t in TYPE_ORDER]
        rm = [np.mean(red[t]) if red[t].size else 0.0 for t in TYPE_ORDER]
        rc = [ci95(red[t]) for t in TYPE_ORDER]
        return cm, cc, rm, rc

    def _bar_labels(ax, means, cis, xs, sign, fs, dx=0.0, ha='center'):
        """Place value labels above (sign +1) or below (sign -1) the bars."""
        for i, (mean, ci) in enumerate(zip(means, cis)):
            if mean > 0 or ci > 0:
                va = 'bottom' if sign > 0 else 'top'
                ax.text(xs[i] + dx, sign * (mean + ci), f"{mean:.1f}", ha=ha,
                        va=va, fontsize=fs,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    if COMBINED:
        # Each traffic keeps its own color; redundant bars get a hatch.
        RED_HATCH = "\\\\\\"
        traffic_colors = {"day_night": tab10_colors[0], "non_station": tab10_colors[2]}
        n = len(TRAFFIC_ORDER)
        width = 0.8 / n

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax.set_axisbelow(True)  # draw grid under the bars

        # Nudge labels so the two adjacent bars' annotations don't overlap.
        label_nudge = width * 0.2

        for j, tr in enumerate(TRAFFIC_ORDER):
            cm, cc, rm, rc = _cold_red(groups[tr])
            offset = (j - (n - 1) / 2) * width
            xs = x + offset
            color = traffic_colors[tr]
            # Cold-hit: positive, no hatch
            ax.bar(xs, cm, width, yerr=cc, ecolor='red', capsize=CAP_SIZE,
                   alpha=0.9, color=color)
            # Redundant: negative, hatched
            ax.bar(xs, [-v for v in rm], width, yerr=rc, ecolor='red',
                   capsize=CAP_SIZE, alpha=0.9, color=color, hatch=RED_HATCH)

            sign_dir = -1 if j < (n - 1) / 2 else (1 if j > (n - 1) / 2 else 0)
            dx = sign_dir * label_nudge
            _bar_labels(ax, cm, cc, xs, 1, BAR_SIZE, dx=dx)
            _bar_labels(ax, rm, rc, xs, -1, BAR_SIZE, dx=dx)

        # Legend: traffic colors + hatch meaning (cold vs redundant)
        legend_handles = [
            Patch(facecolor=traffic_colors[tr], label=TRAFFIC_LABELS[tr])
            for tr in TRAFFIC_ORDER
        ] + [
            Patch(facecolor='white', edgecolor='black', label='Cold-hit'),
            Patch(facecolor='white', edgecolor='black', hatch=RED_HATCH,
                  label='Redundant'),
        ]
        ax.legend(handles=legend_handles, ncol=1)
    else:
        cm, cc, rm, rc = _cold_red(groups)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        ax.set_axisbelow(True)  # draw grid under the bars

        ax.bar(x, cm, yerr=cc, capsize=CAP_SIZE,
               color=tab10_colors[3], label="Cold-hit")
        ax.bar(x, [-v for v in rm], yerr=rc, capsize=CAP_SIZE,
               color=tab10_colors[0], label="Redundant")
        _bar_labels(ax, cm, cc, x, 1, BAR_SIZE)
        _bar_labels(ax, rm, rc, x, -1, BAR_SIZE)
        ax.legend()

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Average per Minute")
    # ax.set_title("Cold-hit vs Redundant warm functions by strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Increase maximum and minimum Y limits slightly to make room for text
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min * 1.15, y_max * 1.15)

    plt.tight_layout()
    plt.show()


def plot_time_series(groups):
    target_types = ["mine",  "RL"]
    # Keep only types that actually have data — RL is currently produced
    # for day_night traffic only, so it is absent for non_station.
    present = [t for t in target_types if groups.get(t)]
    if not present:
        return

    ts_rc = {
        "lines.linewidth":      TS_LINE_WIDTH,
        "axes.labelsize":       TS_LABEL_SIZE,
        "axes.titlesize":       TS_TITLE_SIZE,
        "xtick.labelsize":      TS_TICK_SIZE,
        "ytick.labelsize":      TS_TICK_SIZE,
        "legend.fontsize":      TS_LEGEND_SIZE,
        "legend.title_fontsize": TS_LEGEND_TITLE_SIZE,
    }
    with plt.rc_context(ts_rc):
        tab10_colors = plt.get_cmap("tab10").colors
        n = len(present)
        fig, axes = plt.subplots(n, 1, figsize=(11, 4.5 * n), sharex=True)
        if n == 1:
            axes = [axes]

        has_plotted = False
        legend_handles = []
        for i, t in enumerate(present):
            ax = axes[i]
            rep_idx = 0
            for f in groups[t]:
                for rep_df in load_repetitions(f):
                    rep_df = rep_df.iloc[: len(rep_df) // 2]
                    df = rep_df[rep_df["minute"] >= START_MINUTE]

                    # Only add the label once across the whole figure for the legend
                    first = (i == 0 and rep_idx == 0)
                    p99_label = "p99 Active Function" if first else "_nolegend_"
                    warm_label = "Warm Pool Size" if first else "_nolegend_"

                    l1, = ax.plot(df["minute"], df["p99_active"], color=tab10_colors[0], alpha=0.7, label=p99_label, linewidth=TS_LINE_WIDTH)
                    l2, = ax.plot(df["minute"], df["warm_pool_size"], color=tab10_colors[1], alpha=0.7, label=warm_label, linewidth=TS_LINE_WIDTH)
                    if first:
                        legend_handles = [l1, l2]
                    has_plotted = True
                    rep_idx += 1

            ax.set_ylabel("Count")
            ax.set_title(f"{TYPE_LABELS.get(t, t)}")
            ax.grid(True, linestyle="--", alpha=0.5)

        if not has_plotted:
            plt.close(fig)
            return

        axes[-1].set_xlabel("Time (minutes)")

        fig.legend(handles=legend_handles, ncol=2, loc="lower center",
                   bbox_to_anchor=(0.45, 1.1), bbox_transform=axes[0].transAxes,
                   frameon=True)

        plt.tight_layout()
        plt.show()


def main():
    if COMBINED:
        groups = {tr: discover_files(DIRS[tr]) for tr in TRAFFIC_ORDER}
        print("Combined mode — reading both traffic folders:")
        for tr in TRAFFIC_ORDER:
            print(f"  [{TRAFFIC_LABELS[tr]}] {DIRS[tr]}")
            for t in TYPE_ORDER:
                n_reps = sum(len(load_repetitions(f)) for f in groups[tr][t])
                print(f"    {TYPE_LABELS[t]:12s} ({t}): "
                      f"{len(groups[tr][t])} file(s), {n_reps} repetition(s)")
    else:
        groups = discover_files(LOGS_DIR)
        print(f"Reading logs from: {LOGS_DIR}")
        print("Files / repetitions per type:")
        for t in TYPE_ORDER:
            n_reps = sum(len(load_repetitions(f)) for f in groups[t])
            print(f"  {TYPE_LABELS[t]:12s} ({t}): "
                  f"{len(groups[t])} file(s), {n_reps} repetition(s)")

    plot_blocking(groups)
    # plot_power(groups)
    # plot_ram(groups)
    plot_ram_second_per_request(groups)
    # plot_cpu_second_per_request(groups)
    plot_energy_per_request(groups)
    # plot_ram_per_request(groups)
    plot_latency(groups)
    plot_cold_hit_percentage(groups)

    # plot_cold_redundant handles combined mode itself; plot_time_series
    # still expects single-traffic groups.
    # plot_cold_redundant(groups)
    # if COMBINED:
    #     for tr in TRAFFIC_ORDER:
    #         pass
    #         plot_time_series(groups[tr])
    # else:
    #     pass
    #     plot_time_series(groups)


if __name__ == "__main__":
    main()
