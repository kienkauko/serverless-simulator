#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_csv_R2.py — extends compare_csv.py with per-scenario cold-start time.

For each stable-state window the script measures the empirical cold-start
time as the mean of (response_time_s - processing_time_s) over requests
where warm == False.  This measured value replaces the static SPAWN_TIME_S
when calling the CTMC for that scenario.  If the stable window contains no
cold-start requests (i.e. warm% is high enough that the window is fully
warm), the measured cold-start time is NaN and the static SPAWN_TIME_S
fallback is used instead.
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

# Allow import from repo root (serverless-simulator/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from Markov.model_2D import MarkovModel

# ============================================================
# APPLICATION PROFILE — same as compare_csv.py
# ============================================================
SERVICE_TIME_S     = 2.12
SPAWN_TIME_S       = 11.91          # static fallback when no cold starts observed
SPAWN_DISTRIBUTION = "exponential"

CPU_WARM           = 0.01
RAM_WARM           = 2.60
CPU_DEMAND         = 7.03
RAM_DEMAND         = 2.62
CPU_TRANSIT        = 5.35
RAM_TRANSIT        = 1.51
PEAK_POWER         = 150.0
POWER_SCALE        = 0.2

MU         = 1.0 / SERVICE_TIME_S
SPAWN_RATE = 1.0 / SPAWN_TIME_S

# Stable-state window: middle portion of request stream
LOWER_CUTOFF = 0.0
UPPER_CUTOFF = 0.9

CI_LEVEL = 0.95
N_BOOTSTRAP = 10000

# ============================================================
# File discovery (adapted from analyze_multi.py)
# ============================================================

FILE_RE = re.compile(
    r"^(?P<datetime>\d{4}_\d{2}_\d{2}_\d{4})"
    r"_pod_(?P<pod>\d+)"
    r"_warm_(?P<warm>\d+)"
    r"_rep_(?P<rep>\d+)"
    r"_(?P<kind>requests|resource)\.csv$"
)

ARRIVAL_RATES = [0.25, 0.5, 1.0]


def discover_scenarios(base_dir):
    """Scan request/ and resource/ folders.

    Returns dict:  (arrival_rate, pod, warm%) -> list of (rep, req_path, res_path)
    where warm% is an integer (0-100).
    """
    groups = defaultdict(list)

    for rate in ARRIVAL_RATES:
        # Try both "arrival_1" and "arrival_1.0" style folder names
        candidates = {str(rate), str(rate).rstrip("0").rstrip(".")}
        req_dir = res_dir = None
        for label in candidates:
            rd = os.path.join(base_dir, "data", "request", f"arrival_{label}")
            sd = os.path.join(base_dir, "data", "resource", f"arrival_{label}")
            if os.path.isdir(rd) and os.path.isdir(sd):
                req_dir, res_dir = rd, sd
                break
        if req_dir is None:
            continue

        # Index files by (pod, warm, rep) -> {kind: path}
        lookup = defaultdict(dict)
        for d, kind in [(req_dir, "requests"), (res_dir, "resource")]:
            for path in glob.glob(os.path.join(d, "*.csv")):
                m = FILE_RE.match(os.path.basename(path))
                if not m:
                    continue
                key = (int(m.group("pod")), int(m.group("warm")), int(m.group("rep")))
                lookup[key][kind] = path

        for (pod, warm, rep), files in sorted(lookup.items()):
            if "requests" in files and "resource" in files:
                groups[(rate, pod, warm)].append((rep, files["requests"], files["resource"]))

    return groups


# ============================================================
# CSV loaders  (from analyze_multi.py)
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
            rows.append({
                "time": float(r["time"]),
                "cpu": float(r["cpu"]),
                "ram_pct": float(r["ram_pct"]),
                "ram_mb": float(r["ram_mb"]),
                "serving_requests": float(r["serving_requests"]),
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


def compute_baseline(req_path, res_path):
    requests = load_requests(req_path)
    resource = load_resource(res_path)
    first_request_time = min(r["send_time"] for r in requests)
    res_baseline = [r for r in resource if r["time"] < first_request_time]
    if not res_baseline:
        return 0.0, 0.0
    return (
        time_weighted_mean(res_baseline, "cpu"),
        time_weighted_mean(res_baseline, "ram_pct"),
    )


# ============================================================
# Per-rep metric extraction
# ============================================================

def analyze_rep(req_path, res_path, baseline_cpu=None, baseline_ram_pct=None):
    """Return per-rep metrics dict, or None on failure.

    Adds 'cold_start_time_s': mean (response_time_s - processing_time_s) for
    stable-window requests where warm == False.  NaN when none exist.
    """
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

    response_times   = [r["response_time_s"]   for r in successful_window]
    processing_times = [r["processing_time_s"] for r in successful_window]

    # Cold-start time: overhead incurred when a request triggers a cold start
    cold_start_times = [
        r["response_time_s"] - r["processing_time_s"]
        for r in successful_window
        if not r["warm"]
    ]

    res_window = [r for r in resource if t_start <= r["time"] <= t_end]

    if baseline_cpu is None or baseline_ram_pct is None:
        first_request_time = min(r["send_time"] for r in requests)
        res_baseline = [r for r in resource if r["time"] < first_request_time]
        baseline_cpu     = time_weighted_mean(res_baseline, "cpu") if res_baseline else 0.0
        baseline_ram_pct = time_weighted_mean(res_baseline, "ram_pct") if res_baseline else 0.0

    avg_cpu     = time_weighted_mean(res_window, "cpu")
    avg_ram_pct = time_weighted_mean(res_window, "ram_pct")
    avg_serving = time_weighted_mean(res_window, "serving_requests")

    # Subtract baseline
    avg_cpu     = avg_cpu - baseline_cpu
    avg_ram_pct = avg_ram_pct - baseline_ram_pct

    result = {
        "blocking_rate": blocking_rate,
        "mean_cpu_pct": avg_cpu,
        "mean_ram_pct": avg_ram_pct,
        "cold_start_time_s": float(np.mean(cold_start_times)) if cold_start_times else float("nan"),
    }

    if response_times:
        result["response_time_s"] = float(np.mean(response_times))
    if processing_times:
        result["processing_p99_s"]  = float(np.percentile(processing_times, 99))
        result["processing_var_s2"] = float(np.var(processing_times))

    return result


# ============================================================
# CTMC  (accepts per-scenario spawn_rate)
# ============================================================

def run_ctmc(arrival_rate, warm_percent, total_pods, spawn_rate=None):
    if spawn_rate is None:
        spawn_rate = SPAWN_RATE

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
        return {
            'blocking_probability': metrics['blocking_ratios'][0],
            'latency':              metrics['latency'][0],
            'cpu_usage':            metrics['cpu_usage'][0],
            'ram_usage':            metrics['ram_usage'][0],
            'p99_latency':          metrics['p99_latency'][0],
            'variance_latency':     metrics['variance_latency'][0],
        }
    except Exception as e:
        print(f"  [ERROR] CTMC failed for lam={arrival_rate}, "
              f"warm={warm_percent*100:.0f}%, pods={total_pods}: {e}")
        return None


# ============================================================
# Aggregate per-scenario experiment metrics from reps
# ============================================================

def aggregate_reps(rep_results):
    """Given list of per-rep dicts, return scenario-level means and bootstrap CIs."""
    n = len(rep_results)
    if n == 0:
        return None

    agg = {}
    for key in ['blocking_rate', 'response_time_s', 'mean_cpu_pct', 'mean_ram_pct',
                'processing_p99_s', 'processing_var_s2', 'cold_start_time_s']:
        vals = [r[key] for r in rep_results if key in r and not np.isnan(r[key])]
        if not vals:
            agg[key] = float('nan')
            continue
        vals = np.array(vals, dtype=float)
        m = float(np.mean(vals))
        agg[key] = m
        if len(vals) > 1:
            ci_low, ci_high = _bootstrap_ci(vals, np.mean)
            hw = (ci_high - ci_low) / 2.0
        else:
            ci_low = ci_high = m
            hw = 0.0
        agg[f'{key}_ci_low']  = ci_low
        agg[f'{key}_ci_high'] = ci_high
        agg[f'{key}_ci_hw']   = hw
        agg[f'{key}_n']       = len(vals)

    return agg


# ============================================================
# Error metrics with 90 % CIs  (following CTMC_vs_simulator_multi_core.py)
# ============================================================

def _bootstrap_ci(data, stat_func, n_boot=N_BOOTSTRAP, ci_level=CI_LEVEL, seed=42):
    """Percentile bootstrap CI for a scalar statistic."""
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
    """Compute MAPE, RMSE, R², and bootstrap CIs on MAPE and experiment mean."""
    ctmc_vals = np.array(ctmc_vals, dtype=float)
    exp_vals  = np.array(exp_vals,  dtype=float)
    n = len(ctmc_vals)

    if n == 0:
        return None

    errors = ctmc_vals - exp_vals
    denom = np.where(exp_vals > 1e-6, exp_vals, 1.0)
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
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("CTMC vs Per-Rep CSV Experiment Comparison  [R2: empirical spawn time]")
    print("=" * 70)
    print(f"  Service time:            {SERVICE_TIME_S} s  (mu = {MU:.4f} req/s)")
    print(f"  Static spawn time:       {SPAWN_TIME_S} s  (fallback when no cold starts observed)")
    print(f"  Spawn distribution:      {SPAWN_DISTRIBUTION}")
    print(f"  CPU warm / demand:       {CPU_WARM} % / {CPU_DEMAND} %")
    print(f"  RAM warm / demand:       {RAM_WARM} % / {RAM_DEMAND} %")
    print(f"  Stable-state window:     {LOWER_CUTOFF*100:.0f}th – {UPPER_CUTOFF*100:.0f}th percentile")
    print(f"  CI level:                {CI_LEVEL*100:.0f}%")
    print()

    # --- Discover scenarios --------------------------------------------------
    print("Scanning request/ and resource/ folders...")
    scenarios = discover_scenarios(script_dir)
    if not scenarios:
        print("No matching CSV files found. Exiting.")
        return

    sorted_keys = sorted(scenarios.keys())
    print(f"  Found {len(sorted_keys)} scenarios\n")

    # --- Build per-pod baselines from warm=0 reps ----------------------------
    pod_baselines = {}
    for (rate, pod, warm), reps in scenarios.items():
        if warm != 0:
            continue
        b_cpus, b_rams = [], []
        for _rep, req_path, res_path in reps:
            bc, br = compute_baseline(req_path, res_path)
            b_cpus.append(bc)
            b_rams.append(br)
        if b_cpus:
            pod_baselines[(rate, pod)] = (float(np.mean(b_cpus)), float(np.mean(b_rams)))

    # --- Process each scenario -----------------------------------------------
    ctmc_blocking, exp_blocking = [], []
    ctmc_latency,  exp_latency  = [], []
    ctmc_cpu,      exp_cpu      = [], []
    ctmc_ram,      exp_ram      = [], []
    ctmc_p99,      exp_p99      = [], []
    ctmc_var,      exp_var      = [], []

    # --- First pass: print cold-start times per scenario ---------------------
    print("COLD-START TIME SUMMARY  (mean response_time_s - processing_time_s")
    print("                          for warm==False requests in stable window)")
    print(f"  {'Arrival':>8} {'Warm%':>6} {'Pods':>5}  {'ColdStart_s':>12}  {'N_cold_reps':>12}  {'Source':>8}")
    print("  " + "-" * 60)
    cold_start_summary = {}
    for (rate, pod, warm) in sorted_keys:
        reps = scenarios[(rate, pod, warm)]
        baseline = pod_baselines.get((rate, pod))
        b_cpu, b_ram = baseline if baseline is not None else (None, None)
        rep_results = []
        for _rep, req_path, res_path in sorted(reps):
            result = analyze_rep(req_path, res_path,
                                 baseline_cpu=b_cpu, baseline_ram_pct=b_ram)
            if result is not None:
                rep_results.append(result)
        if not rep_results:
            continue
        agg = aggregate_reps(rep_results)
        if agg is None:
            continue
        cold_start_summary[(rate, pod, warm)] = (agg, rep_results)
        measured = agg.get('cold_start_time_s', float('nan'))
        n_cold = agg.get('cold_start_time_s_n', 0)
        if not np.isnan(measured) and measured > 0:
            source = "measured"
            cs_str = f"{measured:.4f}"
        else:
            source = "static"
            cs_str = f"{'nan':>12} -> {SPAWN_TIME_S:.4f}"
        print(f"  {rate:>8.2f} {warm:>5d}% {pod:>5d}  {cs_str:>12}  {n_cold:>12}  {source:>8}")
    print()

    # --- Main comparison table -----------------------------------------------
    hdr = (f"{'Arrival':>8} {'Warm%':>6} {'Pods':>5} {'Reps':>5}  "
           f"{'SpawnT_s':>9}  "
           f"{'Block_CTMC':>11} {'Block_Exp':>10}  "
           f"{'Lat_CTMC':>9} {'Lat_Exp':>8}  "
           f"{'P99_CTMC':>9} {'P99_Exp':>8}  "
           f"{'Var_CTMC':>9} {'Var_Exp':>8}  "
           f"{'CPU_CTMC':>9} {'CPU_Exp':>8}  "
           f"{'RAM_CTMC':>9} {'RAM_Exp':>8}")
    print(hdr)
    print("-" * len(hdr))

    for (rate, pod, warm) in sorted_keys:
        if (rate, pod, warm) not in cold_start_summary:
            continue
        agg, rep_results = cold_start_summary[(rate, pod, warm)]
        warm_frac = warm / 100.0

        # Determine spawn time for this scenario
        measured_cold_start = agg.get('cold_start_time_s', float('nan'))
        if not np.isnan(measured_cold_start) and measured_cold_start > 0:
            scenario_spawn_time = measured_cold_start
            spawn_source = "measured"
        else:
            scenario_spawn_time = SPAWN_TIME_S
            spawn_source = "static"
        scenario_spawn_rate = 1.0 / scenario_spawn_time

        # Run CTMC with the scenario-specific spawn rate
        ctmc = run_ctmc(rate, warm_frac, pod, spawn_rate=scenario_spawn_rate)
        if ctmc is None:
            continue

        # Experiment values (scenario means)
        exp_block = agg.get('blocking_rate', float('nan'))
        exp_lat   = agg.get('response_time_s', float('nan'))
        exp_cpu_v = agg.get('mean_cpu_pct', float('nan'))
        exp_ram_v = agg.get('mean_ram_pct', float('nan'))
        exp_p99_v = agg.get('processing_p99_s', float('nan'))
        exp_var_v = agg.get('processing_var_s2', float('nan'))

        # Collect pairs
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
        if not np.isnan(exp_p99_v):
            ctmc_p99.append(ctmc['p99_latency'])
            exp_p99.append(exp_p99_v)
        if not np.isnan(exp_var_v):
            ctmc_var.append(ctmc['variance_latency'])
            exp_var.append(exp_var_v)

        spawn_label = f"{scenario_spawn_time:.3f}{'*' if spawn_source == 'static' else ''}"
        print(f"{rate:>8.2f} {warm:>5d}% {pod:>5d} {len(rep_results):>5d}  "
              f"{spawn_label:>9}  "
              f"{ctmc['blocking_probability']:>11.4f} {exp_block:>10.4f}  "
              f"{ctmc['latency']:>9.4f} {exp_lat:>8.4f}  "
              f"{ctmc['p99_latency']:>9.4f} {exp_p99_v:>8.4f}  "
              f"{ctmc['variance_latency']:>9.4f} {exp_var_v:>8.4f}  "
              f"{ctmc['cpu_usage']:>9.3f} {exp_cpu_v:>8.3f}  "
              f"{ctmc['ram_usage']:>9.3f} {exp_ram_v:>8.3f}")

    print("  (* SpawnT_s marked with * used the static fallback; others are measured)")

    # --- Error analysis with 90% CIs ----------------------------------------
    print("\n" + "=" * 70)
    print(f"ERROR ANALYSIS  (CTMC vs Experiment,  {CI_LEVEL*100:.0f}% bootstrap CIs, B={N_BOOTSTRAP})")
    print("Note: blocking probability is compared in 0-1 scale")
    print("=" * 70)

    comparisons = [
        (ctmc_blocking, exp_blocking, 'Blocking probability (0-1 scale)'),
        (ctmc_latency,  exp_latency,  'Latency / response time  [s]'),
        (ctmc_p99,      exp_p99,      'P99 processing time  [s]'),
        (ctmc_var,      exp_var,      'Variance of processing time  [s^2]'),
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
