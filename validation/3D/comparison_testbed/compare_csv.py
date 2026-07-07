#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the 3D dynamic-pool CTMC (Markov/model_3D.py) against per-repetition
measurement CSVs produced by the testbed and parsed the same way as
analyze_multi.py.

Data layout (relative to this file):
    data/request/arrival_<rate>/<...>_pod_<n>_idle_<t>_rep_<k>_requests.csv
    data/resource/arrival_<rate>/<...>_pod_<n>_idle_<t>_rep_<k>_resource.csv

Each scenario is (arrival_rate, total_pods, idle_timeout). The `idle_<t>` value
is the warm-container idle timeout in seconds, so the CTMC warm-timeout rate is
theta = 1 / idle_timeout (idle_0.0 => reaped effectively instantly). The reps for
a scenario are processed independently; the rep means form the experiment
estimate, and CTMC-vs-experiment error metrics (MAPE, RMSE, R^2) are computed
across scenarios with bootstrap confidence intervals.
"""

import csv
import os
import re
import sys
import glob
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# --- make the repo root importable (this file is validation/3D/comparison_testbed) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from Markov.model_3D import MarkovModel

# ============================================================
# APPLICATION PROFILE — measured per-container demands / timings for this testbed
# ============================================================
SERVICE_TIME_S     = 0.45
SPAWN_TIME_S       = 5.35

CPU_WARM           = 0.00
RAM_WARM           = 2.05
CPU_DEMAND         = 4.83
RAM_DEMAND         = 2.16
CPU_TRANSIT        = 6.44
RAM_TRANSIT        = 1.02
PEAK_POWER         = 150.0
POWER_SCALE        = 0.2

MU         = 1.0 / SERVICE_TIME_S
SPAWN_RATE = 1.0 / SPAWN_TIME_S

# idle_0.0 means "reap immediately"; model it as a tiny timeout -> very large theta.
ZERO_IDLE_TIMEOUT_S = 1e-3

# Stable-state window: middle portion of request stream
LOWER_CUTOFF = 0.25
UPPER_CUTOFF = 0.75

# Baseline window: seconds from the start of each resource file (empty system).
BASELINE_WINDOW_S = 20.0

CI_LEVEL = 0.95
N_BOOTSTRAP = 10000

ARRIVAL_RATES = [0.5, 1.0]

# ============================================================
# File discovery (mirrors analyze_multi.py)
# ============================================================

FILE_RE = re.compile(
    r"^(?P<datetime>\d{4}_\d{2}_\d{2}_\d{4})"
    r"_pod_(?P<pod>\d+)"
    r"_idle_(?P<idle>[\d.]+)"
    r"_rep_(?P<rep>\d+)"
    r"_(?P<kind>requests|resource)\.csv$"
)


def discover_scenarios(data_dir):
    """Scan data/request and data/resource folders.

    Returns dict:  (arrival_rate, pod, idle) -> list of (rep, req_path, res_path)
    where idle is the idle-timeout in seconds (float).
    """
    groups = defaultdict(list)

    for rate in ARRIVAL_RATES:
        folder = f"arrival_{rate}"
        req_dir = os.path.join(data_dir, "request", folder)
        res_dir = os.path.join(data_dir, "resource", folder)
        if not (os.path.isdir(req_dir) and os.path.isdir(res_dir)):
            continue

        lookup = defaultdict(dict)
        for d, kind in [(req_dir, "requests"), (res_dir, "resource")]:
            for path in glob.glob(os.path.join(d, "*.csv")):
                m = FILE_RE.match(os.path.basename(path))
                if not m:
                    continue
                key = (int(m.group("pod")), float(m.group("idle")), int(m.group("rep")))
                lookup[key][kind] = path

        for (pod, idle, rep), files in sorted(lookup.items()):
            if "requests" in files and "resource" in files:
                groups[(rate, pod, idle)].append((rep, files["requests"], files["resource"]))

    return groups


# ============================================================
# CSV loaders  (mirror analyze_multi.py)
# ============================================================

def load_requests(path):
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
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # 3D schema: time,cpu,ram_pct,ram_mb,idle,processing,starting
            processing = float(r["processing"])
            starting = float(r["starting"])
            rows.append({
                "time": float(r["time"]),
                "cpu": float(r["cpu"]),
                "ram_pct": float(r["ram_pct"]),
                "ram_mb": float(r["ram_mb"]),
                "serving_requests": processing + starting,
            })
    return rows


def time_weighted_mean(rows, key):
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


def compute_baseline(resource):
    """Baseline = mean over the first BASELINE_WINDOW_S of the resource file."""
    if not resource:
        return 0.0, 0.0
    t0 = resource[0]["time"]
    res_baseline = [r for r in resource if r["time"] <= t0 + BASELINE_WINDOW_S]
    if not res_baseline:
        return 0.0, 0.0
    return (
        time_weighted_mean(res_baseline, "cpu"),
        time_weighted_mean(res_baseline, "ram_pct"),
    )


# ============================================================
# Per-rep metric extraction
# ============================================================

def analyze_rep(req_path, res_path):
    """Return per-rep metrics dict, or None on failure."""
    requests = load_requests(req_path)
    resource = load_resource(res_path)

    total_requests = len(requests)
    if total_requests == 0:
        return None

    failed = sum(1 for r in requests if not r["success"])
    blocking_rate = failed / total_requests

    # Stable-state time window
    send_times = sorted(r["send_time"] for r in requests)
    n = len(send_times)
    t_start = send_times[int(n * LOWER_CUTOFF)]
    t_end   = send_times[min(int(n * UPPER_CUTOFF), n - 1)]

    window_requests = [r for r in requests if t_start <= r["send_time"] <= t_end]
    successful_window = [r for r in window_requests if r["success"]]

    response_times = [r["response_time_s"] for r in successful_window]

    res_window = [r for r in resource if t_start <= r["time"] <= t_end]

    baseline_cpu, baseline_ram_pct = compute_baseline(resource)

    avg_cpu     = time_weighted_mean(res_window, "cpu") - baseline_cpu
    avg_ram_pct = time_weighted_mean(res_window, "ram_pct") - baseline_ram_pct

    result = {
        "blocking_rate": blocking_rate,
        "mean_cpu_pct": avg_cpu,
        "mean_ram_pct": avg_ram_pct,
    }
    if response_times:
        result["response_time_s"] = float(np.mean(response_times))
    return result


# ============================================================
# CTMC (3D dynamic-pool model)
# ============================================================

def run_ctmc(arrival_rate, idle_timeout, total_pods):
    theta = 1.0 / (idle_timeout if idle_timeout > 0 else ZERO_IDLE_TIMEOUT_S)

    config = {
        "arrival_rate":    arrival_rate,
        "service_rate":    MU,
        "spawn_rate":      SPAWN_RATE,
        "theta":           theta,
        "max_queue":       total_pods,
        "ram_warm":        RAM_WARM,
        "cpu_warm":        CPU_WARM,
        "ram_demand":      RAM_DEMAND,
        "cpu_demand":      CPU_DEMAND,
        "cpu_transit":     CPU_TRANSIT,
        "ram_transit":     RAM_TRANSIT,
        "power_max":       PEAK_POWER,
        "power_min_scale": POWER_SCALE,
    }

    try:
        model = MarkovModel(config, verbose=False)
        metrics = model.get_metrics()
        return {
            'blocking_probability': metrics['blocking_ratios'][0],
            'latency':              metrics['latency'][0],
            'cpu_usage':            metrics['cpu_usage'][0],
            'ram_usage':            metrics['ram_usage'][0],
        }
    except Exception as e:
        print(f"  [ERROR] CTMC failed for lam={arrival_rate}, "
              f"idle={idle_timeout}s, pods={total_pods}: {e}")
        return None


# ============================================================
# Aggregate per-scenario experiment metrics from reps
# ============================================================

def aggregate_reps(rep_results):
    """Given list of per-rep dicts, return scenario-level means."""
    if not rep_results:
        return None
    agg = {}
    for key in ['blocking_rate', 'response_time_s', 'mean_cpu_pct', 'mean_ram_pct']:
        vals = [r[key] for r in rep_results if key in r]
        if vals:
            agg[key] = float(np.mean(vals))
    return agg


# ============================================================
# Error metrics with bootstrap CIs
# ============================================================

def _bootstrap_ci(data, stat_func, n_boot=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = stat_func(data[idx])
    alpha = (1 - ci_level) / 2
    return float(np.percentile(boot_stats, alpha * 100)), \
           float(np.percentile(boot_stats, (1 - alpha) * 100))


def calculate_comparison_metrics(ctmc_vals, exp_vals):
    ctmc_vals = np.array(ctmc_vals, dtype=float)
    exp_vals  = np.array(exp_vals,  dtype=float)
    n = len(ctmc_vals)
    if n == 0:
        return None

    errors = ctmc_vals - exp_vals
    denom = np.where(np.abs(exp_vals) > 1e-6, exp_vals, 1.0)
    per_case_ape = np.abs(errors / denom) * 100

    mape = float(np.mean(per_case_ape))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((exp_vals - np.mean(exp_vals)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    if n > 1:
        ci_mape_low, ci_mape_high = _bootstrap_ci(per_case_ape, np.mean)
        ci_exp_low, ci_exp_high = _bootstrap_ci(exp_vals, np.mean)
    else:
        ci_mape_low = ci_mape_high = mape
        ci_exp_low = ci_exp_high = float(exp_vals[0])

    return {
        'MAPE':          mape,
        'RMSE':          rmse,
        'R_squared':     r2,
        'n_samples':     n,
        'ci_mape_low':   ci_mape_low,
        'ci_mape_high':  ci_mape_high,
        'exp_mean':      float(np.mean(exp_vals)),
        'ci_exp_low':    ci_exp_low,
        'ci_exp_high':   ci_exp_high,
        'ctmc_mean':     float(np.mean(ctmc_vals)),
    }


# ============================================================
# Main
# ============================================================

def main():
    data_dir = os.path.join(_HERE, "data")

    print("=" * 70)
    print("3D CTMC vs Per-Rep CSV Experiment Comparison")
    print("=" * 70)
    print(f"  Service time:       {SERVICE_TIME_S} s  (mu = {MU:.4f} req/s)")
    print(f"  Spawn time:         {SPAWN_TIME_S} s  (spawn_rate = {SPAWN_RATE:.4f} /s)")
    print(f"  CPU warm/active/transit: {CPU_WARM} / {CPU_DEMAND} / {CPU_TRANSIT} %")
    print(f"  RAM warm/active/transit: {RAM_WARM} / {RAM_DEMAND} / {RAM_TRANSIT} %")
    print(f"  Stable-state window: {LOWER_CUTOFF*100:.0f}th – {UPPER_CUTOFF*100:.0f}th percentile")
    print(f"  CI level:           {CI_LEVEL*100:.0f}%")
    print()

    print("Scanning data/request and data/resource folders...")
    scenarios = discover_scenarios(data_dir)
    if not scenarios:
        print("No matching CSV files found. Exiting.")
        return

    sorted_keys = sorted(scenarios.keys())
    print(f"  Found {len(sorted_keys)} scenarios\n")

    ctmc_blocking, exp_blocking = [], []
    ctmc_latency,  exp_latency  = [], []
    ctmc_cpu,      exp_cpu      = [], []
    ctmc_ram,      exp_ram      = [], []

    hdr = (f"{'Arrival':>8} {'Idle(s)':>8} {'Pods':>5} {'Reps':>5}  "
           f"{'Block_CTMC':>11} {'Block_Exp':>10}  "
           f"{'Lat_CTMC':>9} {'Lat_Exp':>8}  "
           f"{'CPU_CTMC':>9} {'CPU_Exp':>8}  "
           f"{'RAM_CTMC':>9} {'RAM_Exp':>8}")
    print(hdr)
    print("-" * len(hdr))

    for (rate, pod, idle) in sorted_keys:
        reps = scenarios[(rate, pod, idle)]

        rep_results = []
        for _rep, req_path, res_path in sorted(reps):
            result = analyze_rep(req_path, res_path)
            if result is not None:
                rep_results.append(result)
        if not rep_results:
            continue

        agg = aggregate_reps(rep_results)
        if agg is None:
            continue

        ctmc = run_ctmc(rate, idle, pod)
        if ctmc is None:
            continue

        exp_block = agg.get('blocking_rate', float('nan'))
        exp_lat   = agg.get('response_time_s', float('nan'))
        exp_cpu_v = agg.get('mean_cpu_pct', float('nan'))
        exp_ram_v = agg.get('mean_ram_pct', float('nan'))

        if not np.isnan(exp_block):
            ctmc_blocking.append(ctmc['blocking_probability'])
            exp_blocking.append(exp_block)
        if not np.isnan(exp_lat):
            ctmc_latency.append(ctmc['latency'])
            exp_latency.append(exp_lat)
        if not np.isnan(exp_cpu_v):
            ctmc_cpu.append(ctmc['cpu_usage'])
            exp_cpu.append(exp_cpu_v)
        if not np.isnan(exp_ram_v):
            ctmc_ram.append(ctmc['ram_usage'])
            exp_ram.append(exp_ram_v)

        print(f"{rate:>8.2f} {idle:>8.1f} {pod:>5d} {len(rep_results):>5d}  "
              f"{ctmc['blocking_probability']:>11.4f} {exp_block:>10.4f}  "
              f"{ctmc['latency']:>9.4f} {exp_lat:>8.4f}  "
              f"{ctmc['cpu_usage']:>9.3f} {exp_cpu_v:>8.3f}  "
              f"{ctmc['ram_usage']:>9.3f} {exp_ram_v:>8.3f}")

    print("\n" + "=" * 70)
    print(f"ERROR ANALYSIS  (CTMC vs Experiment,  {CI_LEVEL*100:.0f}% bootstrap CIs, B={N_BOOTSTRAP})")
    print("Note: blocking probability is compared in 0-1 scale")
    print("=" * 70)

    comparisons = [
        (ctmc_blocking, exp_blocking, 'Blocking probability (0-1 scale)'),
        (ctmc_latency,  exp_latency,  'Latency / response time  [s]'),
        (ctmc_cpu,      exp_cpu,      'Mean CPU usage  [%]'),
        (ctmc_ram,      exp_ram,      'Mean RAM usage  [%]'),
    ]

    ci_pct = int(CI_LEVEL * 100)
    for ctmc_v, exp_v, label in comparisons:
        res = calculate_comparison_metrics(ctmc_v, exp_v)
        if res is None:
            print(f"\n  {label}: no data")
            continue
        print(f"\n  {label}  (n = {res['n_samples']})")
        print(f"    Exp mean:     {res['exp_mean']:.6f}   "
              f"{ci_pct}% CI: [{res['ci_exp_low']:.6f}, {res['ci_exp_high']:.6f}]")
        print(f"    CTMC mean:    {res['ctmc_mean']:.6f}")
        print(f"    MAPE:         {res['MAPE']:.2f}%   "
              f"{ci_pct}% CI: [{res['ci_mape_low']:.2f}%, {res['ci_mape_high']:.2f}%]")
        print(f"    RMSE:         {res['RMSE']:.6f}")
        print(f"    R-squared:    {res['R_squared']:.4f}")


if __name__ == '__main__':
    main()
