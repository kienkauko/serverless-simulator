import csv
import re
import os
import glob
from datetime import datetime
import numpy as np
from collections import defaultdict

# ── Configuration ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

ARRIVAL_RATES = [0.25, 0.5, 1.0]

# Percentile cutoffs for stable-state window (fraction, not percent)
LOWER_CUTOFF = 0.25
UPPER_CUTOFF = 0.75

# Regex to parse filenames
FILE_RE = re.compile(
    r"^(?P<datetime>\d{4}_\d{2}_\d{2}_\d{4})"
    r"_pod_(?P<pod>\d+)"
    r"_warm_(?P<warm>\d+)"
    r"_rep_(?P<rep>\d+)"
    r"_(?P<kind>requests|resource)\.csv$"
)


def arrival_dir(arrival_rate):
    """Return (request_dir, resource_dir) for a given arrival rate."""
    folder = f"arrival_{arrival_rate}"
    return (
        os.path.join(DATA_DIR, "request", folder),
        os.path.join(DATA_DIR, "resource", folder),
    )


def discover_files(request_dir, resource_dir):
    """Scan directories and group files by (pod, warm) -> list of (rep, req_path, res_path)."""
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
    """Analyze a single repetition. Returns dict of metrics."""
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
    cold_start_times = [
        r["response_time_s"] - r["processing_time_s"]
        for r in successful_window if not r["warm"]
    ]

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
        "cold_start_times": cold_start_times,
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


def analyze_arrival(arrival_rate, lower_cutoff=LOWER_CUTOFF, upper_cutoff=UPPER_CUTOFF):
    """Analyze all data for a single arrival rate. Returns list of output lines."""
    request_dir, resource_dir = arrival_dir(arrival_rate)
    groups = discover_files(request_dir, resource_dir)

    output_lines = []

    # Build per-pod baseline from warm=0 reps only
    pod_baselines = {}
    for (pod, warm), reps in groups.items():
        if warm != 0:
            continue
        b_cpus, b_ram_pcts, b_ram_mbs = [], [], []
        for _rep, req_path, res_path in reps:
            b_cpu, b_ram_pct, b_ram_mb = compute_baseline(req_path, res_path)
            b_cpus.append(b_cpu)
            b_ram_pcts.append(b_ram_pct)
            b_ram_mbs.append(b_ram_mb)
        if b_cpus:
            pod_baselines[pod] = (
                float(np.mean(b_cpus)),
                float(np.mean(b_ram_pcts)),
                float(np.mean(b_ram_mbs)),
            )

    for (pod, warm), reps in sorted(groups.items()):
        baseline = pod_baselines.get(pod)
        b_cpu, b_ram_pct, b_ram_mb = baseline if baseline is not None else (None, None, None)
        rep_results = []
        for rep, req_path, res_path in sorted(reps):
            result = analyze_rep(req_path, res_path, lower_cutoff, upper_cutoff,
                                 baseline_cpu=b_cpu,
                                 baseline_ram_pct=b_ram_pct,
                                 baseline_ram_mb=b_ram_mb)
            if result is not None:
                rep_results.append(result)

        if not rep_results:
            continue

        blocking_rates = [r["blocking_rate"] for r in rep_results]
        all_response_times = [t for r in rep_results for t in r["response_times"]]
        all_processing_times = [t for r in rep_results for t in r["processing_times"]]
        all_cold_start_times = [t for r in rep_results for t in r["cold_start_times"]]
        avg_servings = [r["avg_serving"] for r in rep_results]
        cpu_per_reqs = [r["cpu_per_req"] for r in rep_results]
        ram_pct_per_reqs = [r["ram_pct_per_req"] for r in rep_results]
        ram_mb_per_reqs = [r["ram_mb_per_req"] for r in rep_results]
        avg_cpus = [r["avg_cpu"] for r in rep_results]
        avg_ram_pcts = [r["avg_ram_pct"] for r in rep_results]

        output_lines.append(f"══ Pod: {pod}, Arrival: {arrival_rate}, Warm: {warm}% ({len(rep_results)} reps) ══")
        output_lines.append(f"  Blocking rate     :  mean={np.mean(blocking_rates)*100:.2f}%  std={np.std(blocking_rates)*100:.2f}%")
        if all_response_times:
            output_lines.append(f"  Response time  (s):  mean={np.mean(all_response_times):.2f}  std={np.std(all_response_times):.2f}")
        else:
            output_lines.append(f"  Response time  (s):  no successful requests in window")
        if all_processing_times:
            output_lines.append(f"  Processing time(s):  mean={np.mean(all_processing_times):.4f}  p99={np.percentile(all_processing_times, 99):.4f}  var={np.var(all_processing_times):.4f}")
        else:
            output_lines.append(f"  Processing time(s):  no successful requests in window")
        if all_cold_start_times:
            output_lines.append(f"  Cold-start time(s):  mean={np.mean(all_cold_start_times):.4f}  p99={np.percentile(all_cold_start_times, 99):.4f}  var={np.var(all_cold_start_times):.4f}")
        else:
            output_lines.append(f"  Cold-start time(s):  na")
        output_lines.append(f"  Serving requests  :  mean={np.mean(avg_servings):.2f}  std={np.std(avg_servings):.2f}")
        output_lines.append(f"  CPU/req        (%):  mean={np.mean(cpu_per_reqs):.4f}  std={np.std(cpu_per_reqs):.4f}")
        output_lines.append(f"  RAM/req        (%):  mean={np.mean(ram_pct_per_reqs):.4f}  std={np.std(ram_pct_per_reqs):.4f}")
        output_lines.append(f"  RAM/req       (MB):  mean={np.mean(ram_mb_per_reqs):.4f}  std={np.std(ram_mb_per_reqs):.4f}")
        output_lines.append(f"  Mean CPU usage (%):  mean={np.mean(avg_cpus):.4f}  std={np.std(avg_cpus):.4f}")
        output_lines.append(f"  Mean RAM usage (%):  mean={np.mean(avg_ram_pcts):.4f}  std={np.std(avg_ram_pcts):.4f}")
        output_lines.append("")

    return output_lines


def analyze(arrival_rates=ARRIVAL_RATES, lower_cutoff=LOWER_CUTOFF, upper_cutoff=UPPER_CUTOFF):
    output_lines = []
    output_lines.append(f"Arrival rates : {arrival_rates}")
    output_lines.append(f"Time window   : {lower_cutoff*100:.0f}th – {upper_cutoff*100:.0f}th percentile of requests")
    output_lines.append("")

    for rate in arrival_rates:
        request_dir, resource_dir = arrival_dir(rate)
        output_lines.append(f"── Arrival rate: {rate} req/s ──")
        output_lines.append(f"  Request dir : {request_dir}")
        output_lines.append(f"  Resource dir: {resource_dir}")
        output_lines.append("")
        output_lines.extend(analyze_arrival(rate, lower_cutoff, upper_cutoff))

    output_str = "\n".join(output_lines)
    print(output_str)

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    result_file = os.path.join(result_dir, f"multi_analysis_result_{timestamp}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(output_str + "\n")


if __name__ == "__main__":
    analyze()
