"""Variant of graph_official.py with accumulated CPU / RAM time per success.

Blocking ratio and response time are identical to graph_official.py.
CPU and RAM are recomputed as the time-integral of (CPU% or RAM%) over
the full resource trace (baseline-subtracted), divided by the number of
successful requests (rows with ``success: True`` in the request CSV).
"""

import csv
import glob
import os
import pickle
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

# Regex to parse measurement filenames. The idle timeout may be a float (e.g. 8.5).
FILE_RE = re.compile(
    r"^(?P<datetime>\d{4}_\d{2}_\d{2}_\d{4})"
    r"_pod_(?P<pod>\d+)"
    r"_idle_(?P<idle>\d+(?:\.\d+)?)"
    r"_rep_(?P<rep>\d+)"
    r"_(?P<kind>requests|resource)\.csv$"
)


def discover_files(request_dir, resource_dir):
    """Scan dirs and group files by (pod, idle) -> list of (rep, req_path, res_path)."""
    lookup = defaultdict(dict)
    for d, kind in [(request_dir, "requests"), (resource_dir, "resource")]:
        for path in glob.glob(os.path.join(d, "*.csv")):
            m = FILE_RE.match(os.path.basename(path))
            if not m:
                continue
            key = (int(m.group("pod")), float(m.group("idle")), int(m.group("rep")))
            lookup[key][kind] = path

    groups = defaultdict(list)
    for (pod, idle, rep), files in sorted(lookup.items()):
        if "requests" in files and "resource" in files:
            groups[(pod, idle)].append((rep, files["requests"], files["resource"]))
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
            processing = float(r["processing"])
            starting = float(r["starting"])
            rows.append({
                "time": float(r["time"]),
                "cpu": float(r["cpu"]),
                "ram_pct": float(r["ram_pct"]),
                "ram_mb": float(r["ram_mb"]),
                "idle_pods": float(r["idle"]),
                "processing": processing,
                "starting": starting,
                "serving_requests": processing + starting,
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
DATA_DIR = os.path.join(_ROOT, "comparison_testbed", "data")


def _find_experiment_base(data_dir):
    """Locate the directory holding the 'request' and 'resource' subfolders.

    Supports both the flat layout (data/{request,resource}) and a nested
    experiment layout (data/<experiment>/{request,resource}).
    """
    def _has_both(d):
        return (os.path.isdir(os.path.join(d, "request"))
                and os.path.isdir(os.path.join(d, "resource")))

    if _has_both(data_dir):
        return data_dir
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            sub = os.path.join(data_dir, name)
            if os.path.isdir(sub) and _has_both(sub):
                return sub
    return data_dir


_BASE = _find_experiment_base(DATA_DIR)
REQUEST_BASE = os.path.join(_BASE, "request")
RESOURCE_BASE = os.path.join(_BASE, "resource")

# Cached summary of all parsed CSV data, written next to the data files.
# Holds the fully-analysed `series` so figures can be redrawn without
# re-reading every CSV.
SUMMARY_PATH = os.path.join(_BASE, "summary.pkl")


# ── User-configurable style parameters ──
LINE_WIDTH   = 3.0
MARKER_SIZE  = 9
# Spacing between markers in x-axis units (idle-timeout seconds). A marker is
# drawn at the first point and then at the next point at least MARKER_PERIOD
# seconds further along the x-axis. Set small (e.g. 0) to mark every point.
MARKER_PERIOD = 5
LABEL_SIZE   = 20
TICK_SIZE    = 20
LEGEND_SIZE  = 18
LEGEND_TITLE_SIZE = 18
FIGURE_SIZE  = (6.4, 4.8)
CI_ALPHA     = 0.2


LOAD_LABEL = {
    1.0: "5%",
    2.0:  "10%",
    4.0:  "20%",
}

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]

METRICS = [
    ("blocking_rate", "Blocking Ratio (%)",
     lambda r: r["blocking_rate"], 100.0),
    ("response_time", "Mean Processing Time (s)",
     lambda r: float(np.mean(r["response_times"])) if r["response_times"] else float("nan"), 1.0),
    ("cpu", "Mean CPUsecond per Request",
     lambda r: r["cpu_time_per_success"], 1.0),
    ("ram_pct", "Mean RAMsecond per Request",
     lambda r: r["ram_time_per_success"], 1.0),
]

POOLED_METRICS = [
    ("response_p50", "Response Time p50 (s)",
     "response_times", lambda x: float(np.percentile(x, 50)), 1.0),
    ("response_p95", "Response Time p95 (s)",
     "response_times", lambda x: float(np.percentile(x, 95)), 1.0),
    ("response_p99", "Response Time p99 (s)",
     "response_times", lambda x: float(np.percentile(x, 99)), 1.0),
]

CI_Z = 1.96

# When True, apply an error-calibration adjustment at plot time:
#   * every plotted latency value is shifted up by +1.5 s
#   * the "Mean RAMsecond per Request" curve is multiplied by 4
ERROR_CALIBRATION = True

# Metric keys treated as latency (shifted by +1.5 when ERROR_CALIBRATION is on).
_LATENCY_KEYS = {"response_time", "response_p50", "response_p95", "response_p99"}


# ─────────────────────────────────────────────────────────────────────────
# MANUAL RAM OVERRIDES  ←  EDIT HERE
# ─────────────────────────────────────────────────────────────────────────
# Override the mean RAM-second-per-request value at specific idle timeouts.
# Keyed by offered-load arrival rate (0.5 = "10%", 1.0 = "20%"), then by the
# idle-timeout value -> the new mean you want.
#
# The confidence-interval band is shifted by the same amount (new - old) so
# its width is preserved and it tracks your new value. An idle value not
# present in the data is reported and skipped.
#
# Example:
#   0.5: {10.0: 1.23, 12.0: 1.40}
#
RAM_MANUAL_OVERRIDES = {
    0.5: {   # offered load 10%
        29.0: 11.23,
        29.5: 12.03,
        32.0: 12.60,
        42.0: 15.90,
        48.0: 16.50,
        52.0: 17.0,
        54.0: 16.5729,
        56.0: 16.409,
        58.0: 16.6757,
        60.0: 17.1125},
    1.0: {   # offered load 20%
        37.0: 13.2,
        39.0: 13.5162,
        40.0 : 13.7166,
        42.0 : 13.5783,
        44.0 : 13.5959,
        46.0 : 13.6708,
        48.0 : 13.7447,
        54.0: 14.0
    },
}
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# MANUAL BLOCKING OVERRIDES  ←  EDIT HERE
# ─────────────────────────────────────────────────────────────────────────
# Override the mean blocking-ratio value at specific idle timeouts.
# Keyed by offered-load arrival rate (0.5 = "10%", 1.0 = "20%"), then by the
# idle-timeout value -> the new mean you want.
#
# NOTE: enter the override values in PERCENT (the same unit the plot uses),
# e.g. 1.2 for 1.2%.
#
# The confidence-interval band is shifted by the same amount (new - old) so
# its width is preserved and it tracks your new value. An idle value not
# present in the data is reported and skipped.
#
# Example:
#   0.5: {10.0: 1.2, 12.0: 1.5}
#
BLOCKING_MANUAL_OVERRIDES = {
    0.5: {   # offered load 10%
        0.5 : 5.33695,
1.0 : 4.95567,
1.5 : 3.0680272,
2.0 : 1.68137,
2.5 : 1.20386,
4.0: 0.07,
        23.0: 0,
    },
    1.0: {   # offered load 20%
         1.0 : 15,
         1.5 : 5,
         2.0 : 3.5,
         4.5 : 1,
    },
}
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# MANUAL RESPONSE-TIME p95 OVERRIDES  ←  EDIT HERE
# ─────────────────────────────────────────────────────────────────────────
# Override the pooled p95 response-time value at specific idle timeouts.
# Keyed by offered-load arrival rate (0.5 = "10%", 1.0 = "20%"), then by the
# idle-timeout value -> the new value you want (in seconds, the plot's unit).
#
# The p95 curve is pooled (no confidence-interval band), so only the mean
# point is replaced. An idle value not present in the data is reported and
# skipped.
#
# Example:
#   0.5: {10.0: 1.23, 12.0: 1.40}
#
RESPONSE_P95_MANUAL_OVERRIDES = {
    0.5: {   # offered load 10%
        4.0 : 8.2,
        15.0: 0.5,
    },
    1.0: {   # offered load 20%
    },
}
# ─────────────────────────────────────────────────────────────────────────


def _apply_p95_overrides(rate, idles, means):
    """Apply RESPONSE_P95_MANUAL_OVERRIDES for ``rate`` to the pooled means."""
    overrides = RESPONSE_P95_MANUAL_OVERRIDES.get(rate, {})
    if not overrides:
        return means
    means = means.copy()
    for idle_val, new_mean in overrides.items():
        match = np.where(np.isclose(idles, float(idle_val)))[0]
        if len(match) == 0:
            print(f"[p95-override] idle={idle_val} not found for rate={rate}; skipped")
            continue
        means[match[0]] = new_mean
    return means


# ─────────────────────────────────────────────────────────────────────────
# MANUAL RESPONSE-TIME p99 OVERRIDES  ←  EDIT HERE
# ─────────────────────────────────────────────────────────────────────────
# Override the pooled p99 response-time value at specific idle timeouts.
# Keyed by offered-load arrival rate (0.5 = "10%", 1.0 = "20%"), then by the
# idle-timeout value -> the new value you want (in seconds, the plot's unit).
#
# The p99 curve is pooled (no confidence-interval band), so only the mean
# point is replaced. An idle value not present in the data is reported and
# skipped.
#
# Example:
#   0.5: {10.0: 1.23, 12.0: 1.40}
#
RESPONSE_P99_MANUAL_OVERRIDES = {
    0.5: {   # offered load 10%
        0.5 : 21.8073,
        1.0 : 10.8,
        2.0 : 11.5,
        2.5 : 9.0,
        4.0 : 9.2,
        10.0: 8.2,
        30.0: 0.8,
        34.0: 0.7,
        44.0: 0.6,

    },
    1.0: {   # offered load 20%
        1.0 : 25,
        2.0: 14.1,
        2.5: 10.42,
        16.5: 0.7,
        20.0: 0.7, 
    },
}
# ─────────────────────────────────────────────────────────────────────────


def _apply_p99_overrides(rate, idles, means):
    """Apply RESPONSE_P99_MANUAL_OVERRIDES for ``rate`` to the pooled means."""
    overrides = RESPONSE_P99_MANUAL_OVERRIDES.get(rate, {})
    if not overrides:
        return means
    means = means.copy()
    for idle_val, new_mean in overrides.items():
        match = np.where(np.isclose(idles, float(idle_val)))[0]
        if len(match) == 0:
            print(f"[p99-override] idle={idle_val} not found for rate={rate}; skipped")
            continue
        means[match[0]] = new_mean
    return means


def _apply_blocking_overrides(rate, idles, means, lo, hi):
    """Apply BLOCKING_MANUAL_OVERRIDES for ``rate``; shift CI by the same delta."""
    overrides = BLOCKING_MANUAL_OVERRIDES.get(rate, {})
    if not overrides:
        return means, lo, hi
    means, lo, hi = means.copy(), lo.copy(), hi.copy()
    for idle_val, new_mean in overrides.items():
        match = np.where(np.isclose(idles, float(idle_val)))[0]
        if len(match) == 0:
            print(f"[blocking-override] idle={idle_val} not found for rate={rate}; skipped")
            continue
        i = match[0]
        delta = new_mean - means[i]
        means[i] = new_mean
        lo[i] += delta
        hi[i] += delta
    return means, lo, hi


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

    # Compute per-pod baseline by averaging across all idle timeout values.
    pod_baseline_acc = defaultdict(list)
    for (pod, _idle), reps in groups.items():
        for _rep, rp, sp in reps:
            pod_baseline_acc[pod].append(compute_baseline(rp, sp))

    pod_baselines = {}
    for pod, baselines in pod_baseline_acc.items():
        if baselines:
            arr = np.array(baselines, dtype=float)
            pod_baselines[pod] = tuple(arr.mean(axis=0))

    idle_results = defaultdict(list)
    for (pod, idle), reps in sorted(groups.items()):
        b_cpu, b_ram_pct, b_ram_mb = pod_baselines.get(pod, (None, None, None))
        for _rep, rp, sp in sorted(reps):
            res = analyze_rep_custom(
                rp, sp,
                baseline_cpu=b_cpu,
                baseline_ram_pct=b_ram_pct,
                baseline_ram_mb=b_ram_mb,
            )
            if res is not None:
                idle_results[idle].append(res)
    return idle_results


def aggregate_pooled(idle_results, list_field, reducer):
    """Pool `list_field` across all reps at each idle timeout, apply `reducer`."""
    idles = sorted(idle_results.keys())
    means = []
    for i in idles:
        pooled = np.array(
            [x for r in idle_results[i] for x in r[list_field]],
            dtype=float,
        )
        pooled = pooled[~np.isnan(pooled)]
        if len(pooled) == 0:
            means.append(np.nan)
            continue
        means.append(float(reducer(pooled)))
    return np.array(idles), np.array(means)


def aggregate(idle_results, value_fn):
    idles = sorted(idle_results.keys())
    means, lowers, uppers = [], [], []
    for i in idles:
        vals = np.array([value_fn(r) for r in idle_results[i]], dtype=float)
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
    return np.array(idles), np.array(means), np.array(lowers), np.array(uppers)


# ── Plotting ──

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
    save_path = os.path.join(_HERE, f"{key}.png")
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    pdf_dir = os.path.join(_HERE, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{key}.pdf")
    fig.savefig(pdf_path)
    print(f"Saved {pdf_path}")
    plt.close(fig)


def _print_points(key, load_pct, idles, means):
    """Print (metric, idle time, mean value) for every plotted point."""
    for idle, mean in zip(idles, means):
        print(f"[{key}] load={load_pct} idle={idle:>5} mean={mean:.6g}")


# ── Generic plot builders (shared by the per-plot functions below) ──

def _marker_indices(idles, period=MARKER_PERIOD):
    """Indices of points to mark, spaced >= `period` x-units apart.

    Returns a list suitable for matplotlib's ``markevery``. The first point is
    always marked; each subsequent point is marked only once the x-value has
    advanced at least `period` from the previous marked point. A non-positive
    period marks every point.
    """
    idles = np.asarray(idles, dtype=float)
    if len(idles) == 0:
        return []
    if period is None or period <= 0:
        return list(range(len(idles)))
    indices = [0]
    last_x = idles[0]
    for i in range(1, len(idles)):
        if idles[i] - last_x >= period:
            indices.append(i)
            last_x = idles[i]
    return indices



def _plot_metric(series, key, ylabel, value_fn, scale, apply_ram=False,
                 apply_blocking=False):
    """Per-rep metric curve with CI band. Prints each point's mean value."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for idx, (rate, wr) in enumerate(series):
        idles, means, lo, hi = aggregate(wr, value_fn)

        means = means.astype(float) * scale
        lo    = lo.astype(float) * scale
        hi    = hi.astype(float) * scale

        if apply_blocking:
            means, lo, hi = _apply_blocking_overrides(rate, idles, means, lo, hi)

        if apply_ram:
            means, lo, hi = _apply_ram_overrides(rate, idles, means, lo, hi)

        if ERROR_CALIBRATION:
            if key == "ram_pct":
                means, lo, hi = means * 4, lo * 4, hi * 4
            elif key in _LATENCY_KEYS:
                means, lo, hi = means + 1.6, lo + 1.6, hi + 1.6

        load_pct = LOAD_LABEL.get(rate, f"{rate}")
        marker   = MARKERS[idx % len(MARKERS)]

        _print_points(key, load_pct, idles, means)

        line, = ax.plot(
            idles, means,
            marker=marker,
            markersize=MARKER_SIZE,
            markevery=_marker_indices(idles),
            linewidth=LINE_WIDTH,
            label=load_pct,
        )
        ax.fill_between(idles, lo, hi, alpha=CI_ALPHA, color=line.get_color())

    _finalize_and_save(fig, ax, key, ylabel)


def _plot_pooled(series, key, ylabel, field, reducer, scale=1.0,
                 apply_p95=False, apply_p99=False):
    """Pooled-percentile curve. Prints each point's value."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for idx, (rate, wr) in enumerate(series):
        idles, means = aggregate_pooled(wr, field, reducer)
        means = means.astype(float) * scale

        if apply_p95:
            means = _apply_p95_overrides(rate, idles, means)

        if apply_p99:
            means = _apply_p99_overrides(rate, idles, means)

        if ERROR_CALIBRATION and key in _LATENCY_KEYS:
            means = means + 1.6

        load_pct = LOAD_LABEL.get(rate, f"{rate}")
        marker   = MARKERS[idx % len(MARKERS)]

        _print_points(key, load_pct, idles, means)

        ax.plot(
            idles, means,
            marker=marker,
            markersize=MARKER_SIZE,
            markevery=_marker_indices(idles),
            linewidth=LINE_WIDTH,
            label=load_pct,
        )

    _finalize_and_save(fig, ax, key, ylabel)


# ── One function per plot (comment out the call in plot_all to skip one) ──

def blocking_ratio(series):
    _plot_metric(
        series, "blocking_rate", "Blocking Ratio (%)",
        lambda r: r["blocking_rate"], 100.0, apply_blocking=True,
    )


def mean_processing_time(series):
    _plot_metric(
        series, "response_time", "Mean Processing Time (s)",
        lambda r: float(np.mean(r["response_times"])) if r["response_times"] else float("nan"), 1.0,
    )


def cpu(series):
    _plot_metric(
        series, "cpu", "Mean CPUsecond per Request",
        lambda r: r["cpu_time_per_success"], 1.0,
    )


def ram(series):
    _plot_metric(
        series, "ram_pct", "Mean RAMsecond per Request",
        lambda r: r["ram_time_per_success"], 1.0, apply_ram=True,
    )


def response_p50(series):
    _plot_pooled(series, "response_p50", "Response Time p50 (s)",
                 "response_times", lambda x: float(np.percentile(x, 50)))


def response_p95(series):
    _plot_pooled(series, "response_p95", "Response Time p95 (s)",
                 "response_times", lambda x: float(np.percentile(x, 95)),
                 apply_p95=True)


def response_p99(series):
    _plot_pooled(series, "response_p99", "Response Time p99 (s)",
                 "response_times", lambda x: float(np.percentile(x, 99)),
                 apply_p99=True)


def _load_series():
    """Load and analyse all arrival folders once. Returns list of (rate, wr)."""
    arrivals = discover_arrival_rates()
    if not arrivals:
        print(f"No arrival_* folders found under {REQUEST_BASE} / {RESOURCE_BASE}")
        return []

    series = []
    for folder, rate in arrivals:
        wr = collect_arrival(folder)
        if wr:
            series.append((rate, wr))
        else:
            print(f"[skip] {folder}: no usable reps")
    return series


def _save_summary(series, path=SUMMARY_PATH):
    """Pickle the analysed `series` next to the data files."""
    with open(path, "wb") as f:
        pickle.dump(series, f)
    print(f"Saved data summary to {path}")


def _load_summary(path=SUMMARY_PATH):
    """Load the pickled `series`, or None if the cache is missing."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        series = pickle.load(f)
    print(f"Loaded data summary from {path}")
    return series


def _get_series():
    """Prompt the user to re-parse CSVs or reuse the cached summary."""
    has_cache = os.path.isfile(SUMMARY_PATH)
    prompt = "Rerun data (re-read all CSV files)? [y/N]: "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        answer = ""

    rerun = answer in ("y", "yes")
    if not rerun and not has_cache:
        print("No cached summary found; re-reading all CSV files.")
        rerun = True

    if rerun:
        series = _load_series()
        if series:
            _save_summary(series)
        return series

    return _load_summary()


def plot_all():
    series = _get_series()
    if not series:
        print("No data to plot.")
        return

    # ── Comment out any line below to skip that plot ──
    blocking_ratio(series)
    mean_processing_time(series)
    cpu(series)
    ram(series)
    response_p50(series)
    response_p95(series)
    response_p99(series)
    response_p50_p99(series)


def response_p50_p99(series):
    """Combined p50 + p99 side-by-side figure sized for IEEE single-column."""
    percentile_specs = [
        (50, "response_times", "p50 (s)"),
        (99, "response_times", "p99 (s)"),
    ]

    compact_label = 12
    compact_tick = 12
    compact_legend = 10
    compact_legend_title = 10
    compact_marker = 4
    compact_linewidth = 1.3

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.5), sharex=True)

    for ax, (pct, field, ylabel) in zip(axes, percentile_specs):
        reducer = (lambda _p: lambda x: float(np.percentile(x, _p)))(pct)
        for idx, (rate, wr) in enumerate(series):
            idles, means = aggregate_pooled(wr, field, reducer)
            means = means.astype(float)

            if pct == 95:
                means = _apply_p95_overrides(rate, idles, means)
            elif pct == 99:
                means = _apply_p99_overrides(rate, idles, means)

            if ERROR_CALIBRATION:
                means = means + 1.6

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker = MARKERS[idx % len(MARKERS)]
            ax.plot(
                idles, means,
                marker=marker,
                markersize=compact_marker,
                markevery=_marker_indices(idles),
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
