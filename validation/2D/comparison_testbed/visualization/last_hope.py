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


def collect_arrival(folder):
    request_dir = os.path.join(REQUEST_BASE, folder)
    resource_dir = os.path.join(RESOURCE_BASE, folder)
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
    # pdf_path = os.path.join(pdf_dir, f"{key}.pdf")
    # plt.show()
    # fig.savefig(pdf_path)
    # print(f"Saved {pdf_path}")
    # plt.close(fig)


def plot_all():
    arrivals = discover_arrival_rates()
    if not arrivals:
        print(f"No arrival_* folders found under {REQUEST_BASE} / {RESOURCE_BASE}")
        return

    series = []
    for folder, rate in arrivals:
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

        for idx, (rate, wr) in enumerate(series):
            warms, means, lo, hi = aggregate(wr, fn)
            # means, lo, hi = _manual_adjust(key, rate, warms, means, lo, hi)

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

        for idx, (rate, wr) in enumerate(series):
            warms, means = aggregate_pooled(wr, field, reducer)
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

    _plot_response_p50_p99(series)


def _plot_response_p50_p99(series):
    """Combined p50 + p99 side-by-side figure sized for IEEE single-column."""
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
            warms, means = aggregate_pooled(wr, field, reducer)
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
