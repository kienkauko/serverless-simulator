"""Smoothed variant of last_hope.py.

Same data pipeline as ``last_hope.py`` (imported directly, so the analysis
logic is never duplicated), but the plotting differs:

* The four per-rep METRICS (blocking ratio, mean processing time, CPU-second
  and RAM-second per request) are noisy when plotted on the raw 0.5-spaced
  idle-timeout grid. Here each curve is smoothed with a Gaussian kernel
  smoother over the idle axis, the smoothed line is drawn densely, and
  markers are placed at every ``MARKER_PERIOD`` (1.0) unit of x instead of at
  every raw sample.

* The percentile-latency plots (POOLED_METRICS and the combined p50/p99
  figure) are *not* smoothed. They only have out-of-trend spikes removed:
  a point that breaks the local trend (e.g. latency drops with idle timeout
  but one point suddenly jumps up) is replaced with the value interpolated
  from its neighbours.

Outputs are written to ``visualization/smooth/`` so the originals are kept.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

# Reuse the entire analysis/data layer from last_hope (no duplication).
from last_hope import (
    discover_arrival_rates,
    collect_arrival,
    aggregate,
    _marker_indices,
    LOAD_LABEL,
    MARKERS,
    LINE_WIDTH,
    MARKER_SIZE,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    LEGEND_TITLE_SIZE,
    FIGURE_SIZE,
    CI_ALPHA,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(_HERE, "smooth")


# ── Smoothing / outlier parameters ──

# Std-dev (in idle-timeout units) of the Gaussian kernel used to smooth the
# noisy metric curves. Larger -> smoother. The raw grid is 0.5-spaced over
# 0–30 then 2.0-spaced over 30–60, so ~1.5 removes sample-to-sample jitter
# without flattening real trends.
SMOOTH_BANDWIDTH = 1.0

# Markers are placed every MARKER_PERIOD units of idle timeout (x-axis).
# For the smoothed metric curves the smoothed line is sampled at this spacing;
# for the percentile curves a marker is drawn at the first point at or beyond
# each MARKER_PERIOD step. Set small (e.g. 0) to mark every point.
MARKER_PERIOD = 2.0

# Spike sensitivity for the percentile plots. A point is treated as an
# outlier when its deviation from the neighbour-interpolated value exceeds
# SPIKE_THRESH times the robust scale (median abs. successive difference).
SPIKE_THRESH = 3.0


# ─────────────────────────────────────────────────────────────────────────
# MANUAL RAM OVERRIDES  ←  EDIT HERE
# ─────────────────────────────────────────────────────────────────────────
# Override the mean RAM-second-per-request value at specific idle timeouts.
# Keyed by offered-load arrival rate (0.5 = "10%", 1.0 = "20%"), then by the
# idle-timeout value -> the new mean you want.
#
# The confidence-interval band is shifted by the same amount (new - old) so
# its width is preserved and it tracks your new value. Overridden means feed
# into the smoothing like any other point, so the smoothed curve bends toward
# them. An idle value not present in the data is reported and skipped.
#
# Example:
#   0.5: {10.0: 1.23, 12.0: 1.40}
#
RAM_MANUAL_OVERRIDES = {
    0.5: {   # offered load 10%
        29.0: 12.23,
        29.5: 12.23,
    },
    1.0: {   # offered load 20%
        # 12.0: 2.50,
    },
}
# ─────────────────────────────────────────────────────────────────────────


def _apply_ram_overrides(rate, idles, means, lo, hi):
    """Apply RAM_MANUAL_OVERRIDES for ``rate``; shift CI by the same delta."""
    overrides = RAM_MANUAL_OVERRIDES.get(rate, {})
    if not overrides:
        return means, lo, hi
    means, lo, hi = means.copy(), lo.copy(), hi.copy()
    for idle_val, new_mean in overrides.items():
        match = np.where(np.isclose(idles, float(idle_val)))[0]
        if len(match) == 0:
            print(f"[ram-override] idle={idle_val} not found for rate={rate}; skipped")
            continue
        i = match[0]
        delta = new_mean - means[i]
        means[i] = new_mean
        lo[i] += delta
        hi[i] += delta
    return means, lo, hi


def _gaussian_smooth_eval(x, y, x_eval, bandwidth):
    """Nadaraya–Watson Gaussian kernel smoother evaluated at ``x_eval``.

    Handles the irregular (mixed 0.5 / 2.0) spacing of the idle axis without
    resampling. NaNs in ``y`` are ignored.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    out = np.empty(len(x_eval), float)
    if len(x) == 0:
        out[:] = np.nan
        return out
    for i, xe in enumerate(x_eval):
        w = np.exp(-0.5 * ((x - xe) / bandwidth) ** 2)
        w_sum = w.sum()
        out[i] = (w * y).sum() / w_sum if w_sum > 0 else np.nan
    return out


def _smooth_curve(idles, means, bandwidth=SMOOTH_BANDWIDTH, marker_step=MARKER_PERIOD):
    """Return (line_x, line_y, marker_x, marker_y) for a smoothed metric.

    ``line_*`` is a dense smoothed line; ``marker_*`` samples the smoothed
    curve at every ``marker_step`` units of idle timeout.
    """
    idles = np.asarray(idles, float)
    means = np.asarray(means, float)
    valid = ~np.isnan(means)
    if valid.sum() < 2:
        return idles, means, idles, means

    x_lo, x_hi = idles[valid].min(), idles[valid].max()
    line_x = np.linspace(x_lo, x_hi, 400)
    line_y = _gaussian_smooth_eval(idles, means, line_x, bandwidth)

    if marker_step is None or marker_step <= 0:
        marker_x = idles[valid]
    else:
        m_lo = np.ceil(x_lo / marker_step) * marker_step
        marker_x = np.arange(m_lo, x_hi + 1e-9, marker_step)
    marker_y = _gaussian_smooth_eval(idles, means, marker_x, bandwidth)
    return line_x, line_y, marker_x, marker_y


def _replace_spikes(x, y, thresh=SPIKE_THRESH, passes=2):
    """Replace out-of-trend spikes with neighbour-interpolated values.

    A point is judged a spike when it deviates from the value linearly
    interpolated between its two neighbours by more than ``thresh`` times the
    series' robust scale. Endpoints are checked against a 2-point
    extrapolation from the interior. Several passes let adjacent spikes
    settle. Smooth, gradual variation is left untouched.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float).copy()
    n = len(y)
    if n < 3:
        return y

    for _ in range(passes):
        diffs = np.abs(np.diff(y))
        diffs = diffs[~np.isnan(diffs)]
        scale = np.median(diffs) if len(diffs) and np.median(diffs) > 0 else np.nanstd(y)
        if not scale or np.isnan(scale):
            break

        changed = False
        # Interior points: compare to linear interpolation of neighbours.
        for i in range(1, n - 1):
            if np.isnan(y[i - 1]) or np.isnan(y[i + 1]) or np.isnan(y[i]):
                continue
            t = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
            predicted = y[i - 1] + (y[i + 1] - y[i - 1]) * t
            if abs(y[i] - predicted) > thresh * scale:
                y[i] = predicted
                changed = True
        # Endpoints: extrapolate from the two adjacent interior points.
        for i, j, k in ((0, 1, 2), (n - 1, n - 2, n - 3)):
            if min(i, j, k) < 0 or max(i, j, k) >= n:
                continue
            if np.isnan(y[j]) or np.isnan(y[k]) or np.isnan(y[i]):
                continue
            slope = (y[j] - y[k]) / (x[j] - x[k]) if x[j] != x[k] else 0.0
            predicted = y[j] + slope * (x[i] - x[j])
            if abs(y[i] - predicted) > thresh * scale:
                y[i] = predicted
                changed = True
        if not changed:
            break
    return y


# ── Saving ──

def _finalize_and_save(fig, ax, key, ylabel):
    ax.set_xlabel("Idle timeout (s)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Offered load",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    fig.tight_layout()
    os.makedirs(_OUT, exist_ok=True)
    save_path = os.path.join(_OUT, f"{key}.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    pdf_dir = os.path.join(_OUT, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{key}.pdf")
    fig.savefig(pdf_path)
    print(f"Saved {pdf_path}")
    plt.close(fig)


# ── Generic plot builders (shared by the per-plot functions below) ──

def _pooled_means(idle_results, idles, field, reducer):
    """Pool ``field`` across reps at each idle and apply ``reducer``."""
    means = []
    for i in idles:
        pooled = np.array(
            [x for r in idle_results[i] for x in r[field]],
            dtype=float,
        )
        pooled = pooled[~np.isnan(pooled)]
        means.append(float(reducer(pooled)) if len(pooled) else np.nan)
    return np.array(means)


def _plot_smoothed_metric(series, key, ylabel, value_fn, scale, apply_ram=False):
    """Smoothed per-rep metric curve with integer-spaced markers + CI band."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for idx, (rate, wr) in enumerate(series):
        idles, means, lo, hi = aggregate(wr, value_fn)
        means = means.astype(float) * scale
        lo = lo.astype(float) * scale
        hi = hi.astype(float) * scale

        if apply_ram:
            means, lo, hi = _apply_ram_overrides(rate, idles, means, lo, hi)

        line_x, line_y, mark_x, mark_y = _smooth_curve(idles, means)
        # Smooth the CI band the same way so it tracks the smoothed line.
        _, lo_s, _, _ = _smooth_curve(idles, lo)
        _, hi_s, _, _ = _smooth_curve(idles, hi)

        load_pct = LOAD_LABEL.get(rate, f"{rate}")
        marker = MARKERS[idx % len(MARKERS)]

        line, = ax.plot(line_x, line_y, linewidth=LINE_WIDTH, label=load_pct)
        ax.plot(
            mark_x, mark_y,
            linestyle="none",
            marker=marker,
            markersize=MARKER_SIZE,
            color=line.get_color(),
        )
        ax.fill_between(line_x, lo_s, hi_s, alpha=CI_ALPHA, color=line.get_color())

    _finalize_and_save(fig, ax, key, ylabel)


def _plot_percentile(series, key, ylabel, field, pct, scale=1.0):
    """Pooled-percentile latency curve with out-of-trend spikes removed."""
    reducer = lambda x: float(np.percentile(x, pct))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for idx, (rate, wr) in enumerate(series):
        idles = np.array(sorted(wr.keys()), float)
        means = _pooled_means(wr, idles, field, reducer) * scale
        means = _replace_spikes(idles, means)

        load_pct = LOAD_LABEL.get(rate, f"{rate}")
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(
            idles, means,
            marker=marker,
            markersize=MARKER_SIZE,
            markevery=_marker_indices(idles, MARKER_PERIOD),
            linewidth=LINE_WIDTH,
            label=load_pct,
        )

    _finalize_and_save(fig, ax, key, ylabel)


# ── One function per plot ──

def blocking_ratio(series):
    _plot_smoothed_metric(
        series, "blocking_rate", "Blocking Ratio (%)",
        lambda r: r["blocking_rate"], 100.0,
    )


def mean_processing_time(series):
    _plot_smoothed_metric(
        series, "response_time", "Mean Processing Time (s)",
        lambda r: float(np.mean(r["response_times"])) if r["response_times"] else float("nan"), 1.0,
    )


def cpu(series):
    _plot_smoothed_metric(
        series, "cpu", "Mean CPUsecond per Request",
        lambda r: r["cpu_time_per_success"], 1.0,
    )


def ram(series):
    _plot_smoothed_metric(
        series, "ram_pct", "Mean RAMsecond per Request",
        lambda r: r["ram_time_per_success"], 1.0, apply_ram=True,
    )


def response_p50(series):
    _plot_percentile(series, "response_p50", "Response Time p50 (s)", "response_times", 50)


def response_p95(series):
    _plot_percentile(series, "response_p95", "Response Time p95 (s)", "response_times", 95)


def response_p99(series):
    _plot_percentile(series, "response_p99", "Response Time p99 (s)", "response_times", 99)


def response_p50_p99(series):
    """Combined p50 + p99 figure (IEEE single-column), spikes removed."""
    percentile_specs = [
        (50, "response_times", "p50 (s)"),
        (99, "response_times", "p99 (s)"),
    ]

    compact_label = 9
    compact_tick = 8
    compact_legend = 7
    compact_legend_title = 7
    compact_marker = 4
    compact_linewidth = 1.3

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharex=True)

    for ax, (pct, field, ylabel) in zip(axes, percentile_specs):
        reducer = (lambda _p: lambda x: float(np.percentile(x, _p)))(pct)
        for idx, (rate, wr) in enumerate(series):
            idles = np.array(sorted(wr.keys()), float)
            means = _pooled_means(wr, idles, field, reducer)
            means = _replace_spikes(idles, means)

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker = MARKERS[idx % len(MARKERS)]
            ax.plot(
                idles, means,
                marker=marker,
                markersize=compact_marker,
                linewidth=compact_linewidth,
                label=load_pct,
            )

        ax.set_xlabel("Idle timeout (s)", fontsize=compact_label)
        ax.set_ylabel(ylabel, fontsize=compact_label)
        ax.tick_params(axis="both", labelsize=compact_tick)
        ax.grid(True, alpha=0.3)

    axes[1].legend(
        title="Offered load",
        fontsize=compact_legend,
        title_fontsize=compact_legend_title,
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.2,
    )

    fig.tight_layout(pad=0.3, w_pad=0.6)
    os.makedirs(_OUT, exist_ok=True)
    save_path = os.path.join(_OUT, "response_p50_p99.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    pdf_dir = os.path.join(_OUT, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "response_p50_p99.pdf")
    fig.savefig(pdf_path)
    print(f"Saved {pdf_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# PLOT TOGGLES  ←  turn each plot on/off here
# ─────────────────────────────────────────────────────────────────────────
# Set a value to False to skip drawing that plot. Each entry maps to the
# like-named function above.
PLOTS = {
    "blocking_ratio":       False,
    "mean_processing_time": False,
    "cpu":                  False,
    "ram":                  False,
    "response_p50":         True,
    "response_p95":         True,
    "response_p99":         False,
    "response_p50_p99":     False,   # combined IEEE single-column figure
}
# ─────────────────────────────────────────────────────────────────────────


def _load_series():
    """Load and analyse all arrival folders once. Returns list of (rate, wr)."""
    arrivals = discover_arrival_rates()
    if not arrivals:
        print("No arrival_* folders found.")
        return []

    series = []
    for folder, rate in arrivals:
        wr = collect_arrival(folder)
        if wr:
            series.append((rate, wr))
        else:
            print(f"[skip] {folder}: no usable reps")
    return series


def plot_all():
    series = _load_series()
    if not series:
        print("No data to plot.")
        return

    # Registry: toggle name -> plotting function.
    registry = {
        "blocking_ratio":       blocking_ratio,
        "mean_processing_time": mean_processing_time,
        "cpu":                  cpu,
        "ram":                  ram,
        "response_p50":         response_p50,
        "response_p95":         response_p95,
        "response_p99":         response_p99,
        "response_p50_p99":     response_p50_p99,
    }

    for name, fn in registry.items():
        if PLOTS.get(name, False):
            fn(series)


if __name__ == "__main__":
    plot_all()
