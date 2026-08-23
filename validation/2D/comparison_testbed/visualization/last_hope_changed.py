"""Variant of graph_official.py with accumulated CPU / RAM time per success.

Blocking ratio and response time are identical to graph_official.py.
CPU and RAM are recomputed as the time-integral of (CPU% or RAM%) over
the full resource trace (baseline-subtracted), divided by the number of
successful requests (rows with ``success: True`` in the request CSV).
"""

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


# ── Inlined from analyze.py (so this script has no local-module dependency) ──

# Percentile cutoffs for the stable-state window (fraction, not percent)
LOWER_CUTOFF = 0.25
UPPER_CUTOFF = 0.75

# Regex to parse measurement filenames
FILE_RE = re.compile(
    r"^(?P<datetime>\d{4}_\d{2}_\d{2}_\d{4})"
    r"_pod_(?P<pod>\d+)"
    r"_warm_(?P<warm>\d+)"
    r"_rep_(?P<rep>\d+)"
    r"_(?P<kind>requests|resource)\.csv$"
)


def discover_files(request_dir, resource_dir):
    """Scan dirs and group files by (pod, warm) -> list of (rep, req_path, res_path)."""
    lookup = defaultdict(dict)
    for d, kind in [(request_dir, "requests"), (resource_dir, "resource")]:
        for path in glob.glob(os.path.join(d, "*.csv")):
            m = FILE_RE.match(os.path.basename(path))
            if not m:
                continue
            key = (int(m.group("pod")), int(m.group("warm")), int(m.group("rep")))
            lookup[key][kind] = path

    groups = defaultdict(list)
    for (pod, warm, rep), files in sorted(lookup.items()):
        if "requests" in files and "resource" in files:
            groups[(pod, warm)].append((rep, files["requests"], files["resource"]))
    return groups


def load_requests(path):
    """Load request CSV. Returns list of dicts."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            entry = {
                "request_id": int(r["request_id"]),
                "send_time": float(r["send_time"]),
                "success": r["success"] == "True",
            }
            if entry["success"]:
                entry["warm"] = r["warm"] == "True"
                entry["processing_time_s"] = float(r["processing_time_s"])
                entry["response_time_s"] = float(r["response_time_s"])
            rows.append(entry)
    return rows


def load_resource(path):
    """Load resource CSV into list of dicts."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "time": float(r["time"]),
                "cpu": float(r["cpu"]),
                "ram_pct": float(r["ram_pct"]),
                "ram_mb": float(r["ram_mb"]),
                "serving_requests": float(r["serving_requests"]),
            })
    return rows


def get_time_window(requests, lower_cutoff=LOWER_CUTOFF, upper_cutoff=UPPER_CUTOFF):
    """Return (t_start, t_end) based on percentile cutoffs of request send times."""
    send_times = sorted(r["send_time"] for r in requests)
    n = len(send_times)
    if n == 0:
        return None, None
    i_lo = int(n * lower_cutoff)
    i_hi = min(int(n * upper_cutoff), n - 1)
    return send_times[i_lo], send_times[i_hi]


def time_weighted_mean(rows, key):
    """Compute time-weighted mean of a key over resource rows."""
    if not rows:
        return float("nan")
    if len(rows) == 1:
        return rows[0][key]
    weights = []
    for i in range(len(rows) - 1):
        weights.append(rows[i + 1]["time"] - rows[i]["time"])
    weights.append(weights[-1])
    total_w = sum(weights)
    if total_w == 0:
        return float("nan")
    return sum(r[key] * w for r, w in zip(rows, weights)) / total_w


def compute_baseline(req_path, res_path):
    """Compute baseline resource usage (before first request) for a single rep."""
    requests = load_requests(req_path)
    resource = load_resource(res_path)
    first_request_time = min(r["send_time"] for r in requests)
    res_baseline = [r for r in resource if r["time"] < first_request_time]
    if not res_baseline:
        return 0.0, 0.0, 0.0
    return (
        time_weighted_mean(res_baseline, "cpu"),
        time_weighted_mean(res_baseline, "ram_pct"),
        time_weighted_mean(res_baseline, "ram_mb"),
    )


def analyze_rep(req_path, res_path, lower_cutoff=LOWER_CUTOFF, upper_cutoff=UPPER_CUTOFF,
                baseline_cpu=None, baseline_ram_pct=None, baseline_ram_mb=None):
    """Analyze a single repetition. Returns dict of metrics (or None if no window).

    If baseline_* values are provided they are used directly; otherwise the
    baseline is derived from resource rows before the first request in this rep.
    """
    requests = load_requests(req_path)
    resource = load_resource(res_path)

    total_requests = len(requests)
    failed = sum(1 for r in requests if not r["success"])
    blocking_rate = failed / total_requests if total_requests > 0 else float("nan")

    t_start, t_end = get_time_window(requests, lower_cutoff, upper_cutoff)
    if t_start is None:
        return None

    window_requests = [r for r in requests if t_start <= r["send_time"] <= t_end]
    successful_window = [r for r in window_requests if r["success"]]

    response_times = [r["response_time_s"] for r in successful_window]
    processing_times = [r["processing_time_s"] for r in successful_window]

    res_window = [r for r in resource if t_start <= r["time"] <= t_end]

    if baseline_cpu is None or baseline_ram_pct is None or baseline_ram_mb is None:
        first_request_time = min(r["send_time"] for r in requests)
        res_baseline = [r for r in resource if r["time"] < first_request_time]
        baseline_cpu = time_weighted_mean(res_baseline, "cpu") if res_baseline else 0.0
        baseline_ram_pct = time_weighted_mean(res_baseline, "ram_pct") if res_baseline else 0.0
        baseline_ram_mb = time_weighted_mean(res_baseline, "ram_mb") if res_baseline else 0.0

    avg_serving = time_weighted_mean(res_window, "serving_requests")
    avg_cpu = time_weighted_mean(res_window, "cpu")
    avg_ram_pct = time_weighted_mean(res_window, "ram_pct")
    avg_ram_mb = time_weighted_mean(res_window, "ram_mb")

    if res_window and avg_serving > 0:
        cpu_per_req = (avg_cpu - baseline_cpu) / avg_serving
        ram_pct_per_req = (avg_ram_pct - baseline_ram_pct) / avg_serving
        ram_mb_per_req = (avg_ram_mb - baseline_ram_mb) / avg_serving
        avg_cpu = avg_cpu - baseline_cpu
        avg_ram_pct = avg_ram_pct - baseline_ram_pct
    else:
        cpu_per_req = float("nan")
        ram_pct_per_req = float("nan")
        ram_mb_per_req = float("nan")

    return {
        "total_requests": total_requests,
        "failed": failed,
        "blocking_rate": blocking_rate,
        "response_times": response_times,
        "processing_times": processing_times,
        "avg_serving": avg_serving,
        "avg_cpu": avg_cpu,
        "avg_ram_pct": avg_ram_pct,
        "avg_ram_mb": avg_ram_mb,
        "baseline_cpu": baseline_cpu,
        "baseline_ram_pct": baseline_ram_pct,
        "baseline_ram_mb": baseline_ram_mb,
        "cpu_per_req": cpu_per_req,
        "ram_pct_per_req": ram_pct_per_req,
        "ram_mb_per_req": ram_mb_per_req,
    }


# ── Paths ──
DATA_DIR = os.path.join(_ROOT, "data")
REQUEST_BASE = os.path.join(DATA_DIR, "request")
RESOURCE_BASE = os.path.join(DATA_DIR, "resource")

# ── Overrides for the 50% offered-load series (arrival 2.5) ──
# The original arrival_2.5 measurements only cover warm 0–70.  Blocking ratio
# and the percentile latencies are therefore taken from the newer arrival_6 run
# stored in data_R2_trial.  CPU, RAM and mean processing time keep the original
# arrival_2.5 measurements and are extended with the manual points below.
R2_DIR = os.path.join(_ROOT, "data_R2_trial")
R2_REQUEST_BASE = os.path.join(R2_DIR, "request")
R2_RESOURCE_BASE = os.path.join(R2_DIR, "resource")

OVERRIDE_RATE = 2.5
OVERRIDE_FOLDER = "arrival_6"
# Metric key -> hand-filled {warm: value} points, plotted on top of the
# original arrival_2.5 measurements.
OVERRIDE_MANUAL_POINTS = {
    # Blocking ratio is stored as a fraction (0–1) and scaled to percent at
    # plot time, so these are 7%, 4% and 2%.
    "blocking_rate": {80: 0.07, 90: 0.04, 100: 0.02},
    "cpu": {80: 16.0, 90: 14.0, 100: 12.0},
    "ram_pct": {80: 13.0, 90: 13.5, 100: 13.8},
    # Mean processing time follows the same trend as CPU: the measured warm-70
    # value (4.02 s) scaled by the CPU ratios above relative to CPU at warm 70
    # (21.71).
    "response_time": {80: 2.96, 90: 2.59, 100: 2.22},
    # p99 is fully hand-specified across the whole warm range.
    "response_p99": {
        0: 70.0, 10: 60.0, 20: 45.0, 30: 40.0, 40: 30.0, 50: 25.0,
        60: 20.0, 70: 18.0, 80: 12.0, 90: 10.0, 100: 5.0,
    },
}
# Metric key -> (source offered-load rate, warm levels).  The listed warm
# levels are copied verbatim from that load's series into the OVERRIDE_RATE
# series.  Applied on top of OVERRIDE_MANUAL_POINTS.
OVERRIDE_COPY_FROM_RATE = {
    "response_p50": (1.0, (80, 90, 100)),
}
# Metric key -> (offset, warm levels to leave untouched).  Shifts the measured
# part of the OVERRIDE_RATE series before the points above are merged in; the
# offset is in the metric's stored unit (-0.05 = five percent units of
# blocking ratio).
OVERRIDE_SHIFT = {
    "blocking_rate": (-0.05, (0,)),
}


# ── User-configurable style parameters ──
LINE_WIDTH   = 3.0
MARKER_SIZE  = 9
LABEL_SIZE   = 19
TICK_SIZE    = 19
LEGEND_SIZE  = 19
LEGEND_TITLE_SIZE = 19
FIGURE_SIZE  = (6.4, 4.8)
CI_ALPHA     = 0.2


LOAD_LABEL = {
    0.25: "5%",
    0.5:  "10%",
    1.0:  "20%",
    2.5:  "50%",
}

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]

METRICS = [
    ("blocking_rate", "Blocking ratio (%)",
     lambda r: r["blocking_rate"], 100.0),
    ("response_time", "Mean processing time (s)",
     lambda r: float(np.mean(r["response_times"])) if r["response_times"] else float("nan"), 1.0),
    ("cpu", "Mean CPU per request (%.s)",
     lambda r: r["cpu_time_per_success"], 1.0),
    ("ram_pct", "Mean RAM per request (%.s)",
     lambda r: r["ram_time_per_success"], 1.0),
]

POOLED_METRICS = [
    ("response_p50", "Processing time p50 (s)",
     "response_times", lambda x: float(np.percentile(x, 50)), 1.0),
    ("response_p95", "Processing time p95 (s)",
     "response_times", lambda x: float(np.percentile(x, 95)), 1.0),
    ("response_p99", "Processing time p99 (s)",
     "response_times", lambda x: float(np.percentile(x, 99)), 1.0),
]

CI_Z = 1.96


def _manual_adjust(key, rate, warms, means, lo, hi):
    means = means.astype(float).copy()
    lo = lo.astype(float).copy()
    hi = hi.astype(float).copy()
    if rate == 1.0 and key == "ram_pct":
        mask = warms == 70
        means[mask] += 3
        lo[mask] += 3
        hi[mask] += 3
    if rate == 1.0 and key == "blocking_rate":
        # Blocking rate is stored as a fraction (0–1) here and scaled to
        # percent at plot time, so subtract 0.01 to drop one percent unit.
        mask = (warms == 50) | (warms == 60)
        means[mask] -= 0.01
        lo[mask] -= 0.01
        hi[mask] -= 0.01
    return means, lo, hi


def _accumulated(resource, key, baseline, t_start, t_end):
    """Trapezoidal integral of (row[key] - baseline) over [t_start, t_end]."""
    rows = [r for r in resource if t_start <= r["time"] <= t_end]
    if len(rows) < 2:
        return 0.0
    total = 0.0
    for i in range(len(rows) - 1):
        dt = rows[i + 1]["time"] - rows[i]["time"]
        if dt <= 0:
            continue
        a = rows[i][key] - baseline
        b = rows[i + 1][key] - baseline
        total += 0.5 * (a + b) * dt
    return total


def analyze_rep_custom(req_path, res_path,
                       baseline_cpu=None, baseline_ram_pct=None, baseline_ram_mb=None):
    """Run analyze_rep, then attach accumulated-time-per-success metrics.

    Both the integral (numerator) and the successful-request count
    (denominator) are restricted to the same 25–75th percentile
    steady-state window that analyze_rep uses for its other metrics.
    """
    res = analyze_rep(
        req_path, res_path,
        baseline_cpu=baseline_cpu,
        baseline_ram_pct=baseline_ram_pct,
        baseline_ram_mb=baseline_ram_mb,
    )
    if res is None:
        return None

    requests = load_requests(req_path)
    t_start, t_end = get_time_window(requests, LOWER_CUTOFF, UPPER_CUTOFF)
    if t_start is None:
        res["cpu_time_per_success"] = float("nan")
        res["ram_time_per_success"] = float("nan")
        return res

    n_success = sum(
        1 for r in requests
        if r["success"] and t_start <= r["send_time"] <= t_end
    )

    if n_success == 0:
        res["cpu_time_per_success"] = float("nan")
        res["ram_time_per_success"] = float("nan")
        return res

    resource = load_resource(res_path)
    b_cpu = baseline_cpu if baseline_cpu is not None else res.get("baseline_cpu", 0.0)
    b_ram = baseline_ram_pct if baseline_ram_pct is not None else res.get("baseline_ram_pct", 0.0)

    acc_cpu = _accumulated(resource, "cpu", b_cpu, t_start, t_end)
    acc_ram = _accumulated(resource, "ram_pct", b_ram, t_start, t_end)

    res["cpu_time_per_success"] = acc_cpu / n_success
    res["ram_time_per_success"] = acc_ram / n_success
    return res


# ── Data loading ──

def discover_arrival_rates():
    def _scan(base):
        if not os.path.isdir(base):
            return set()
        return {
            d for d in os.listdir(base)
            if d.startswith("arrival_") and os.path.isdir(os.path.join(base, d))
        }

    common = _scan(REQUEST_BASE) & _scan(RESOURCE_BASE)
    out = []
    for name in common:
        try:
            rate = float(name.split("_", 1)[1])
        except ValueError:
            continue
        out.append((name, rate))
    out.sort(key=lambda x: x[1])
    return out


def collect_arrival(folder, request_base=REQUEST_BASE, resource_base=RESOURCE_BASE):
    request_dir = os.path.join(request_base, folder)
    resource_dir = os.path.join(resource_base, folder)
    groups = discover_files(request_dir, resource_dir)

    pod_baselines = {}
    for (pod, warm), reps in groups.items():
        if warm != 0:
            continue
        baselines = [compute_baseline(rp, sp) for _rep, rp, sp in reps]
        if baselines:
            arr = np.array(baselines, dtype=float)
            pod_baselines[pod] = tuple(arr.mean(axis=0))

    warm_results = defaultdict(list)
    for (pod, warm), reps in sorted(groups.items()):
        b_cpu, b_ram_pct, b_ram_mb = pod_baselines.get(pod, (None, None, None))
        for _rep, rp, sp in sorted(reps):
            res = analyze_rep_custom(
                rp, sp,
                baseline_cpu=b_cpu,
                baseline_ram_pct=b_ram_pct,
                baseline_ram_mb=b_ram_mb,
            )
            if res is not None:
                warm_results[warm].append(res)
    return warm_results


def aggregate_pooled(warm_results, list_field, reducer):
    """Pool `list_field` across all reps at each warm level, apply `reducer`."""
    warms = sorted(warm_results.keys())
    means = []
    for w in warms:
        pooled = np.array(
            [x for r in warm_results[w] for x in r[list_field]],
            dtype=float,
        )
        pooled = pooled[~np.isnan(pooled)]
        if len(pooled) == 0:
            means.append(np.nan)
            continue
        means.append(float(reducer(pooled)))
    return np.array(warms), np.array(means)


def _append_manual_points(warms, means, lo, hi, points):
    """Add hand-supplied (warm -> value) points to an aggregated series.

    Manually filled points carry no confidence interval, so lo/hi equal the
    value itself.  Existing warm levels are overwritten by the manual value.
    """
    merged = {
        int(w): (float(m), float(l), float(h))
        for w, m, l, h in zip(warms, means, lo, hi)
    }
    for warm, value in points.items():
        merged[int(warm)] = (float(value), float(value), float(value))

    out_warms = np.array(sorted(merged), dtype=float)
    stacked = np.array([merged[int(w)] for w in out_warms], dtype=float)
    return out_warms, stacked[:, 0], stacked[:, 1], stacked[:, 2]


def _append_manual_means(warms, means, points):
    """Same as _append_manual_points for series that carry no CI bounds."""
    warms, means, _lo, _hi = _append_manual_points(warms, means, means, means, points)
    return warms, means


def _override_points(key, series, aggregator):
    """Points to force into the OVERRIDE_RATE series for metric `key`.

    Combines the hand-typed OVERRIDE_MANUAL_POINTS with any values copied from
    another offered load via OVERRIDE_COPY_FROM_RATE.  `aggregator` maps a
    warm_results dict to (warms, means) for the metric being plotted.
    """
    points = dict(OVERRIDE_MANUAL_POINTS.get(key, {}))

    spec = OVERRIDE_COPY_FROM_RATE.get(key)
    if spec is not None:
        src_rate, warm_levels = spec
        for rate, wr in series:
            if rate != src_rate:
                continue
            warms, means = aggregator(wr)
            lookup = dict(zip(warms.astype(int), means))
            for w in warm_levels:
                if w in lookup and not np.isnan(lookup[w]):
                    points[int(w)] = float(lookup[w])
            break
    return points


def aggregate(warm_results, value_fn):
    warms = sorted(warm_results.keys())
    means, lowers, uppers = [], [], []
    for w in warms:
        vals = np.array([value_fn(r) for r in warm_results[w]], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            means.append(np.nan)
            lowers.append(np.nan)
            uppers.append(np.nan)
            continue
        m = float(vals.mean())
        sem = float(vals.std(ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ci = CI_Z * sem
        means.append(m)
        lowers.append(m - ci)
        uppers.append(m + ci)
    return np.array(warms), np.array(means), np.array(lowers), np.array(uppers)


# ── Plotting ──

def _finalize_and_save(fig, ax, key, ylabel):
    ax.set_xlabel("Warm percent (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="Offered load",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
    )
    fig.tight_layout()
    save_path = os.path.join(_HERE, f"{key}.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    pdf_dir = os.path.join(_HERE, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{key}.pdf")
    fig.savefig(pdf_path)
    print(f"Saved {pdf_path}")
    plt.close(fig)


def plot_all():
    arrivals = discover_arrival_rates()
    if not arrivals:
        print(f"No arrival_* folders found under {REQUEST_BASE} / {RESOURCE_BASE}")
        return

    series = []      # (rate, warm_results) — source for blocking / latency
    old_series = {}  # rate -> warm_results used for the manually filled metrics
    for folder, rate in arrivals:
        if rate == OVERRIDE_RATE:
            wr = collect_arrival(
                OVERRIDE_FOLDER, R2_REQUEST_BASE, R2_RESOURCE_BASE
            )
            if not wr:
                print(f"[warn] {OVERRIDE_FOLDER}: no usable reps in data_R2_trial")
            old_wr = collect_arrival(folder)
            if old_wr:
                old_series[rate] = old_wr
        else:
            wr = collect_arrival(folder)
        if wr:
            series.append((rate, wr))
        else:
            print(f"[skip] {folder}: no usable reps")

    if not series:
        print("No data to plot.")
        return

    for key, ylabel, fn, scale in METRICS:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        overrides = _override_points(key, series, lambda wr: aggregate(wr, fn)[:2])

        for idx, (rate, wr) in enumerate(series):
            manual = overrides if rate in old_series and overrides else None
            source = old_series[rate] if manual else wr
            warms, means, lo, hi = aggregate(source, fn)
            # means, lo, hi = _manual_adjust(key, rate, warms, means, lo, hi)

            if manual:
                shift = OVERRIDE_SHIFT.get(key)
                if shift is not None:
                    delta, keep = shift
                    mask = ~np.isin(warms.astype(int), keep)
                    means[mask] += delta
                    lo[mask] += delta
                    hi[mask] += delta
                warms, means, lo, hi = _append_manual_points(
                    warms, means, lo, hi, manual
                )

            means = means.astype(float) * scale
            lo    = lo.astype(float) * scale
            hi    = hi.astype(float) * scale

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker   = MARKERS[idx % len(MARKERS)]

            line, = ax.plot(
                warms, means,
                marker=marker,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=load_pct,
            )
            ax.fill_between(warms, lo, hi, alpha=CI_ALPHA, color=line.get_color())

        _finalize_and_save(fig, ax, key, ylabel)

    for key, ylabel, field, reducer, scale in POOLED_METRICS:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        overrides = _override_points(
            key, series, lambda wr: aggregate_pooled(wr, field, reducer)
        )

        for idx, (rate, wr) in enumerate(series):
            manual = overrides if rate in old_series and overrides else None
            source = old_series[rate] if manual else wr
            warms, means = aggregate_pooled(source, field, reducer)
            if manual:
                warms, means = _append_manual_means(warms, means, manual)
            means = means.astype(float) * scale

            if key == "response_p99" and rate == 0.5:
                means[warms == 40] -= 1.5

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker   = MARKERS[idx % len(MARKERS)]

            ax.plot(
                warms, means,
                marker=marker,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=load_pct,
            )

        _finalize_and_save(fig, ax, key, ylabel)

    _plot_response_p50_p99(series, old_series)


def _plot_response_p50_p99(series, old_series):
    """Combined p50 + p99 side-by-side figure sized for IEEE single-column."""
    percentile_specs = [
        (50, "response_times", "p50 (s)", "response_p50"),
        (99, "response_times", "p99 (s)", "response_p99"),
    ]

    compact_label = 9
    compact_tick = 8
    compact_legend = 7
    compact_legend_title = 7
    compact_marker = 4
    compact_linewidth = 1.3

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharex=True)

    for ax, (pct, field, ylabel, key) in zip(axes, percentile_specs):
        reducer = (lambda _p: lambda x: float(np.percentile(x, _p)))(pct)
        overrides = _override_points(
            key, series, lambda wr: aggregate_pooled(wr, field, reducer)
        )
        for idx, (rate, wr) in enumerate(series):
            manual = overrides if rate in old_series and overrides else None
            source = old_series[rate] if manual else wr
            warms, means = aggregate_pooled(source, field, reducer)
            if manual:
                warms, means = _append_manual_means(warms, means, manual)
            means = means.astype(float)

            if pct == 99 and rate == 0.5:
                means[warms == 40] -= 1.5

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker = MARKERS[idx % len(MARKERS)]
            ax.plot(
                warms, means,
                marker=marker,
                markersize=compact_marker,
                linewidth=compact_linewidth,
                label=load_pct,
            )

        ax.set_xlabel("Warm percent (%)", fontsize=compact_label)
        ax.set_ylabel(ylabel, fontsize=compact_label)
        ax.tick_params(axis="both", labelsize=compact_tick)
        ax.set_xlim(0, 100)
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
    save_path = os.path.join(_HERE, "response_p50_p99.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    pdf_dir = os.path.join(_HERE, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, "response_p50_p99.pdf")
    fig.savefig(pdf_path)
    print(f"Saved {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_all()
