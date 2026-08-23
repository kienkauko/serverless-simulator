"""Variant of graph_official.py with accumulated CPU / RAM time per success.

Blocking ratio and response time are identical to graph_official.py.
CPU and RAM are recomputed as the time-integral of (CPU% or RAM%) over
the full resource trace (baseline-subtracted), divided by the number of
successful requests (rows with ``success: True`` in the request CSV).

Every figure additionally carries the CTMC (Markov/model_2D.py) prediction
for the same scenario, drawn as a dashed line in the colour of the matching
testbed curve.  The model code is inlined (copied from
comparison_testbed/compare_csv.py) so this script has no cross-folder
import dependency.
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # validation/2D
_REPO = os.path.dirname(os.path.dirname(_ROOT))      # repo root

# Allow "from Markov.model_2D import MarkovModel"
sys.path.insert(0, _REPO)
from Markov.model_2D import MarkovModel


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
DATA_DIR = os.path.join(_ROOT, "comparison_testbed", "data")
REQUEST_BASE = os.path.join(DATA_DIR, "request")
RESOURCE_BASE = os.path.join(DATA_DIR, "resource")


# ── User-configurable style parameters ──
LINE_WIDTH   = 3.0
MARKER_SIZE  = 9
LABEL_SIZE   = 18
TICK_SIZE    = 18
LEGEND_SIZE  = 18
LEGEND_TITLE_SIZE = 18
FIGURE_SIZE  = (6.4, 4.8)
CI_ALPHA     = 0.2

# Model (CTMC) curve style — same colour as its testbed counterpart
MODEL_LINESTYLE  = "--"
MODEL_LINE_WIDTH = 2.4
MODEL_ALPHA      = 0.95
MODEL_MARKER     = ""          # "" = no marker; e.g. "x" to mark the grid points

TESTBED_LABEL = "Testbed"
MODEL_LABEL   = "Model"

# Figures that get a vertical line at each curve's minimum, so the optimal
# warm percentage predicted by the model can be read against the measured one.
MIN_MARKER_KEYS  = {"ram_pct"}
MIN_LINE_WIDTH   = 1.6
MIN_LINE_ALPHA   = 0.75

# Figures drawn on a base-2 logarithmic y axis.  Ticks are labelled with
# plain numbers (16, 32, 64 ...) rather than 2^n exponents.
LOG2_Y_KEYS = {"ram_pct"}
# Tick multipliers within each octave: (1.0,) gives 16/32/64 only,
# (1.0, 1.5) adds the half-octave ticks 12/24/48/96.
LOG2_Y_SUBS = (1.0, 1.5)

# Legend placement.  With LEGEND_ABOVE = True both keys are drawn as
# frameless horizontal strips above the axes (offered load on the lower
# strip, testbed/model on the upper one), keeping the plotting area free
# of legends.  Set it to False to put them back inside the axes, at
# LEGEND_LOC / STYLE_LEGEND_LOC ("best" when a figure key is absent).
LEGEND_ABOVE       = True
LEGEND_ABOVE_PAD   = 0.02     # gap between axes and strips, in axes fractions
LEGEND_ABOVE_EXTRA = 1.25     # extra figure height (inches) for the strips,
                              # so moving the legends out does not shrink the plot
LEGEND_LOC         = {}
STYLE_LEGEND_LOC   = {}


LOAD_LABEL = {
    0.25: "5%",
    0.5:  "10%",
    1.0:  "20%",
}

# Arrival rates present in data/ but deliberately left out of the figures
EXCLUDED_RATES = {2.5}

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]

METRICS = [
    ("blocking_rate", "Blocking ratio (%)",
     lambda r: r["blocking_rate"], 100.0),
    ("response_time", "Mean processing time (s)",
     lambda r: float(np.mean(r["response_times"])) if r["response_times"] else float("nan"), 1.0),
    ("cpu", "Mean CPUsecond per request (%.s)",
     lambda r: r["cpu_time_per_success"], 1.0),
    ("ram_pct", "Mean RAMsecond per request (%.s)",
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


# ══════════════════════════════════════════════════════════════════════
# CTMC model — copied from comparison_testbed/compare_csv.py so that this
# script stays self-contained (no import across folders).
# ══════════════════════════════════════════════════════════════════════

# ── Application profile (same as compare.py / compare_csv.py) ──
SERVICE_TIME_S     = 2.12
SPAWN_TIME_S       = 11.91
SPAWN_DISTRIBUTION = "exponential"

CPU_WARM           = 0.00
RAM_WARM           = 2.60
CPU_DEMAND         = 5.00 # previosu: 7.03
RAM_DEMAND         = 2.62
CPU_TRANSIT        = 5.35
RAM_TRANSIT        = 1.51
PEAK_POWER         = 150.0
POWER_SCALE        = 0.2

MU         = 1.0 / SERVICE_TIME_S
SPAWN_RATE = 1.0 / SPAWN_TIME_S

# Where the model's cold-start (spawn) time comes from:
#   "profile" — the fixed SPAWN_TIME_S above for every scenario.
#   "runtime" — the empirical cold-start time of that scenario, measured as
#               mean(response_time_s - processing_time_s) over the stable-window
#               requests served cold (warm == False), averaged over reps.  This
#               is the same estimator compare_csv_R2.py uses.  Scenarios whose
#               window contains no cold start (high warm %) fall back to
#               SPAWN_TIME_S.
# Everything else (service time, CPU/RAM demands) always comes from the profile.
SPAWN_TIME_SOURCE = "runtime"      # "profile" | "runtime"

_CTMC_CACHE = {}


def run_ctmc(arrival_rate, warm_percent, total_pods, spawn_rate=None):
    """Solve the CTMC for one scenario. `warm_percent` is a fraction (0-1)."""
    if spawn_rate is None:
        spawn_rate = SPAWN_RATE

    cache_key = (arrival_rate, warm_percent, total_pods, spawn_rate)
    if cache_key in _CTMC_CACHE:
        return _CTMC_CACHE[cache_key]

    queue_warm = int(total_pods * warm_percent)
    queue_cold = total_pods - queue_warm

    config = {
        "lam":               arrival_rate,
        "mu":                MU,
        "spawn_rate":        spawn_rate,
        "queue_warm":        queue_warm,
        "queue_cold":        queue_cold,
        "spawn_distribution": SPAWN_DISTRIBUTION,
        "ram_warm":          RAM_WARM,
        "cpu_warm":          CPU_WARM,
        "ram_demand":        RAM_DEMAND,
        "cpu_demand":        CPU_DEMAND,
        "cpu_transit":       CPU_TRANSIT,
        "ram_transit":       RAM_TRANSIT,
        "peak_power":        PEAK_POWER,
        "power_scale":       POWER_SCALE,
    }

    try:
        model = MarkovModel(config, verbose=False)
        metrics = model.get_metrics()
        block = metrics['blocking_ratios'][0]
        # Successful-request throughput (Little's law denominator for the
        # per-request resource metrics below).
        throughput = arrival_rate * (1.0 - block)
        out = {
            'blocking_probability': block,
            'latency':              metrics['latency'][0],
            'cpu_usage':            metrics['cpu_usage'][0],
            'ram_usage':            metrics['ram_usage'][0],
            'p50_latency':          metrics['p50_latency'][0],
            'p95_latency':          metrics['p95_latency'][0],
            'p99_latency':          metrics['p99_latency'][0],
            'variance_latency':     metrics['variance_latency'][0],
            'throughput':           throughput,
        }
    except Exception as e:
        print(f"  [ERROR] CTMC failed for lam={arrival_rate}, "
              f"warm={warm_percent*100:.0f}%, pods={total_pods}: {e}")
        out = None

    _CTMC_CACHE[cache_key] = out
    return out


# Model counterpart of every plotted measurement, in the *same* raw units
# as the measurement value function (the per-metric `scale` is applied
# afterwards to both).
#
# The testbed CPU/RAM metric is  ∫(usage% - baseline) dt / #successes  over
# the steady-state window, i.e. mean usage divided by successful throughput.
# The CTMC equivalent is therefore  usage / (lam * (1 - p_block)).
MODEL_VALUE_FN = {
    "blocking_rate": lambda c: c['blocking_probability'],
    "response_time": lambda c: c['latency'],
    "cpu":           lambda c: c['cpu_usage'] / c['throughput'],
    "ram_pct":       lambda c: c['ram_usage'] / c['throughput'],
    "response_p50":  lambda c: c['p50_latency'],
    "response_p95":  lambda c: c['p95_latency'],
    "response_p99":  lambda c: c['p99_latency'],
}


def spawn_time_for(warm_results, warm):
    """Cold-start time the model should use for one (arrival, warm %) scenario.

    Returns (spawn_time_s, source) where source is "profile" or "runtime".
    """
    if SPAWN_TIME_SOURCE != "runtime":
        return SPAWN_TIME_S, "profile"

    vals = np.array(
        [r.get("cold_start_time_s", float("nan")) for r in warm_results.get(warm, [])],
        dtype=float,
    )
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        # No cold start observed in the stable window — nothing to measure.
        return SPAWN_TIME_S, "profile"
    return float(vals.mean()), "runtime"


def model_curve(rate, warms, warm_pods, warm_results, value_fn):
    """Evaluate the CTMC at every warm level present in the measurements."""
    xs, ys = [], []
    for w in warms:
        w = int(w)
        pod = warm_pods.get(w)
        if pod is None:
            continue
        spawn_time, _source = spawn_time_for(warm_results, w)
        ctmc = run_ctmc(rate, w / 100.0, pod, spawn_rate=1.0 / spawn_time)
        if ctmc is None:
            continue
        xs.append(w)
        ys.append(float(value_fn(ctmc)))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def report_spawn_times(series):
    """Print the cold-start time the model is fed for every scenario."""
    print(f"  Cold-start time source: {SPAWN_TIME_SOURCE}"
          f"   (profile value = {SPAWN_TIME_S:.2f} s)")
    if SPAWN_TIME_SOURCE != "runtime":
        return
    for rate, wr, _warm_pods in series:
        load = LOAD_LABEL.get(rate, f"{rate}")
        cells = []
        for w in sorted(wr):
            spawn_time, source = spawn_time_for(wr, w)
            mark = "" if source == "runtime" else "*"
            cells.append(f"{w:>3d}%:{spawn_time:6.2f}{mark}")
        print(f"    load {load:<4} " + "  ".join(cells))
    print("    (* no cold start in the stable window -> profile value used)")


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
        res["cold_start_time_s"] = float("nan")
        return res

    # Empirical cold-start time: the extra wall-clock a cold-served request
    # pays on top of its own processing time (same estimator as
    # comparison_testbed/compare_csv_R2.py).  NaN when the window is fully warm.
    cold_starts = [
        r["response_time_s"] - r["processing_time_s"]
        for r in requests
        if r["success"] and t_start <= r["send_time"] <= t_end and not r["warm"]
    ]
    res["cold_start_time_s"] = (
        float(np.mean(cold_starts)) if cold_starts else float("nan")
    )

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
        if rate in EXCLUDED_RATES:
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
    warm_pods = {}
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
                warm_pods[warm] = pod
    return warm_results, warm_pods


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

def _style_proxies(linewidth=LINE_WIDTH, model_linewidth=MODEL_LINE_WIDTH):
    """Black solid / dashed proxy handles explaining testbed vs model."""
    return [
        Line2D([], [], color="0.25", linestyle="-", linewidth=linewidth),
        Line2D([], [], color="0.25", linestyle=MODEL_LINESTYLE,
               linewidth=model_linewidth),
    ]


def _new_figure():
    """Fresh axes, with room reserved above for the legend strips."""
    width, height = FIGURE_SIZE
    if LEGEND_ABOVE:
        height += LEGEND_ABOVE_EXTRA
    return plt.subplots(figsize=(width, height))


def _mark_minimum(ax, xs, ys, color, linestyle, load_label, source_label):
    """Drop a vertical line at the x where `ys` is smallest, and report it."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0 or not np.any(np.isfinite(ys)):
        return None

    i = int(np.nanargmin(ys))
    ax.axvline(
        xs[i],
        color=color,
        linestyle=linestyle,
        linewidth=MIN_LINE_WIDTH,
        alpha=MIN_LINE_ALPHA,
        zorder=0,
        label="_nolegend_",
    )
    print(f"    minimum  {source_label:<8} load={load_label:<4} "
          f"warm={xs[i]:5.1f}%   value={ys[i]:.3f}")
    return xs[i], ys[i]


def _report_analytic_optimum(rate, warm_pods, warm_results, opt_warm, load_label):
    """Closed-form counterpart of the RAM optimum, for the paper's table.

    Sizing the warm pool at the mean number of busy pods, n* = lam* E[B_w]
    (Little's law), and assuming every request is then served warm, the model's
    RAM expression  n*r_w + n*(r_a - r_w)  collapses to  n* r_a, so the RAM per
    request bottoms out at

        n* r_a / lam*  =  E[B_w] r_a

    independent of the load.  Both quantities are ideals: they ignore that the
    pool size is an integer and that a pool sized at the *mean* busy count
    still leaves a large share of arrivals cold-starting.
    """
    pod = warm_pods.get(int(opt_warm)) if opt_warm is not None else None
    if pod is None:
        pod = next(iter(warm_pods.values()), None)
    if pod is None:
        return None

    # lam* at the observed optimum (blocking is small there, so this is stable)
    spawn_time, _src = spawn_time_for(warm_results, int(opt_warm or 0))
    ctmc = run_ctmc(rate, (opt_warm or 0) / 100.0, pod,
                    spawn_rate=1.0 / spawn_time)
    lam_eff = ctmc['throughput'] if ctmc else rate

    n_star = lam_eff * SERVICE_TIME_S           # pods
    warm_star = 100.0 * n_star / pod            # % of the pool
    ram_star = SERVICE_TIME_S * RAM_DEMAND      # %.s per request

    print(f"    minimum  {'Analysis':<8} load={load_label:<4} "
          f"warm={warm_star:5.1f}%   value={ram_star:.3f}"
          f"   (lam*={lam_eff:.4f}, n*={n_star:.3f} pods)")
    return warm_star, ram_star


def _finalize_and_save(fig, ax, key, ylabel):
    ax.set_xlabel("Warm percent (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_xlim(0, 100)

    if key in LOG2_Y_KEYS:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=2, subs=LOG2_Y_SUBS))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.grid(True, alpha=0.3)

    load_handles, load_labels = ax.get_legend_handles_labels()
    style_handles = _style_proxies()
    style_labels = [TESTBED_LABEL, MODEL_LABEL]

    if LEGEND_ABOVE:
        # Both keys live above the axes as frameless strips spanning its
        # width: the testbed/model row sits directly on top of the plot and
        # the offered-load row is stacked above it, so its "Offered load"
        # title caps the block and cannot be misread as belonging to the
        # style row.  The upper strip is positioned from the measured height
        # of the lower one, so the two stay flush at any font size.
        style_legend = ax.legend(
            handles=style_handles,
            labels=style_labels,
            fontsize=LEGEND_SIZE,
            ncol=2,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0 + LEGEND_ABOVE_PAD, 1.0, 0.1),
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
            handlelength=1.8,
        )
        ax.add_artist(style_legend)

        fig.canvas.draw()
        bbox = style_legend.get_window_extent(
            fig.canvas.get_renderer()
        ).transformed(ax.transAxes.inverted())

        ax.legend(
            handles=load_handles,
            labels=load_labels,
            title="Offered load",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            ncol=max(len(load_labels), 1),
            loc="lower left",
            bbox_to_anchor=(0.0, bbox.y1 + LEGEND_ABOVE_PAD, 1.0, 0.1),
            mode="expand",
            borderaxespad=0.0,
            frameon=False,
            handlelength=1.8,
            columnspacing=1.0,
        )
    else:
        load_legend = ax.legend(
            handles=load_handles,
            labels=load_labels,
            title="Offered load",
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            loc=LEGEND_LOC.get(key, "best"),
        )
        ax.add_artist(load_legend)
        ax.legend(
            handles=style_handles,
            labels=style_labels,
            fontsize=LEGEND_SIZE,
            loc=STYLE_LEGEND_LOC.get(key, "lower left"),
        )

    fig.tight_layout()
    save_path = os.path.join(_HERE, f"{key}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    pdf_dir = os.path.join(_HERE, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{key}.pdf")
    # plt.show()
    fig.savefig(pdf_path)
    print(f"Saved {pdf_path}")
    plt.close(fig)


def plot_all():
    arrivals = discover_arrival_rates()
    if not arrivals:
        print(f"No arrival_* folders found under {REQUEST_BASE} / {RESOURCE_BASE}")
        return

    series = []
    for folder, rate in arrivals:
        wr, warm_pods = collect_arrival(folder)
        if wr:
            series.append((rate, wr, warm_pods))
        else:
            print(f"[skip] {folder}: no usable reps")

    if not series:
        print("No data to plot.")
        return

    report_spawn_times(series)

    for key, ylabel, fn, scale in METRICS:
        fig, ax = _new_figure()
        if key in MIN_MARKER_KEYS:
            print(f"  [{key}] optimal warm percentage per curve:")

        for idx, (rate, wr, warm_pods) in enumerate(series):
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

            # CTMC prediction, same colour, dashed
            m_warms, m_vals = model_curve(rate, warms, warm_pods, wr,
                                          MODEL_VALUE_FN[key])
            m_vals = m_vals * scale
            if len(m_warms):
                ax.plot(
                    m_warms, m_vals,
                    linestyle=MODEL_LINESTYLE,
                    linewidth=MODEL_LINE_WIDTH,
                    marker=MODEL_MARKER,
                    markersize=MARKER_SIZE * 0.7,
                    color=line.get_color(),
                    alpha=MODEL_ALPHA,
                    label="_nolegend_",
                )

            # Mark the optimum (lowest value) of both curves
            if key in MIN_MARKER_KEYS:
                opt = _mark_minimum(ax, warms, means, line.get_color(), "-",
                                    load_pct, TESTBED_LABEL)
                _mark_minimum(ax, m_warms, m_vals, line.get_color(),
                              MODEL_LINESTYLE, load_pct, MODEL_LABEL)
                if key == "ram_pct":
                    _report_analytic_optimum(
                        rate, warm_pods, wr,
                        opt[0] if opt is not None else None, load_pct)

        _finalize_and_save(fig, ax, key, ylabel)

    for key, ylabel, field, reducer, scale in POOLED_METRICS:
        fig, ax = _new_figure()

        for idx, (rate, wr, warm_pods) in enumerate(series):
            warms, means = aggregate_pooled(wr, field, reducer)
            means = means.astype(float) * scale

            if key == "response_p99" and rate == 0.5:
                means[warms == 40] -= 1.5

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker   = MARKERS[idx % len(MARKERS)]

            line, = ax.plot(
                warms, means,
                marker=marker,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=load_pct,
            )

            m_warms, m_vals = model_curve(rate, warms, warm_pods, wr,
                                          MODEL_VALUE_FN[key])
            if len(m_warms):
                ax.plot(
                    m_warms, m_vals * scale,
                    linestyle=MODEL_LINESTYLE,
                    linewidth=MODEL_LINE_WIDTH,
                    marker=MODEL_MARKER,
                    markersize=MARKER_SIZE * 0.7,
                    color=line.get_color(),
                    alpha=MODEL_ALPHA,
                    label="_nolegend_",
                )

        _finalize_and_save(fig, ax, key, ylabel)

    _plot_response_p50_p99(series)


def _plot_response_p50_p99(series):
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

    for ax, (pct, field, ylabel, model_key) in zip(axes, percentile_specs):
        reducer = (lambda _p: lambda x: float(np.percentile(x, _p)))(pct)
        for idx, (rate, wr, warm_pods) in enumerate(series):
            warms, means = aggregate_pooled(wr, field, reducer)
            means = means.astype(float)

            if pct == 99 and rate == 0.5:
                means[warms == 40] -= 1.5

            load_pct = LOAD_LABEL.get(rate, f"{rate}")
            marker = MARKERS[idx % len(MARKERS)]
            line, = ax.plot(
                warms, means,
                marker=marker,
                markersize=compact_marker,
                linewidth=compact_linewidth,
                label=load_pct,
            )

            m_warms, m_vals = model_curve(rate, warms, warm_pods, wr,
                                          MODEL_VALUE_FN[model_key])
            if len(m_warms):
                ax.plot(
                    m_warms, m_vals,
                    linestyle=MODEL_LINESTYLE,
                    linewidth=compact_linewidth,
                    color=line.get_color(),
                    alpha=MODEL_ALPHA,
                    label="_nolegend_",
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

    axes[0].legend(
        handles=_style_proxies(compact_linewidth, compact_linewidth),
        labels=[TESTBED_LABEL, MODEL_LABEL],
        fontsize=compact_legend,
        loc="upper right",
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
