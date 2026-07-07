#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the 3D dynamic-pool CTMC (Markov/model_3D.py) predictions against the
analyzed experiment results produced by analyze_multi.py and stored as txt files
in the result/ sub-folder.

Each result txt contains sections like:
    ══ Pod: 10, Arrival: 0.5, Idle: 0.0 (10 reps) ══
      Blocking rate     :  mean=12.02%  std=8.66%
      Response time  (s):  mean=8.78  std=3.36
      Processing time(s):  p99=0.7417  var=0.0072
      Serving requests  :  mean=4.18  std=0.55
      Eff. arrival (r/s):  mean=0.4376  std=0.0895
      CPU/req     (%·s) :  mean=46.7562  std=2.6198
      RAM/req     (%·s) :  mean=11.5530  std=0.8423
      RAM/req    (MB·s) :  mean=3702.0930  std=267.5972
      Mean CPU usage (%):  mean=20.2907  std=3.6065
      Mean RAM usage (%):  mean=5.0200  std=0.9662

`Idle: <t>` is the warm-container idle timeout in seconds, so the CTMC warm-timeout
rate is theta = 1 / idle_timeout (Idle: 0.0 => reaped effectively instantly).
"""

import os
import re
import sys
import numpy as np
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
SERVICE_TIME_S     = 0.45        # mean service time per request (seconds)
SPAWN_TIME_S       = 5.35        # mean cold-start / spawn time (seconds)

CPU_WARM           = 0.00        # % CPU consumed by one idle warm pod
RAM_WARM           = 2.05        # % RAM consumed by one idle warm pod
CPU_DEMAND         = 4.83        # % CPU consumed when pod is actively serving
RAM_DEMAND         = 2.16        # % RAM consumed when pod is actively serving
CPU_TRANSIT        = 6.44        # % CPU consumed during cold-start transition
RAM_TRANSIT        = 1.02        # % RAM consumed during cold-start transition
PEAK_POWER         = 150.0       # Watts — server peak power
POWER_SCALE        = 0.2         # idle power as fraction of peak power

MU         = 1.0 / SERVICE_TIME_S
SPAWN_RATE = 1.0 / SPAWN_TIME_S

# Idle: 0.0 means "reap immediately"; model it as a tiny timeout -> very large theta.
ZERO_IDLE_TIMEOUT_S = 1e-3
# ============================================================


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def parse_txt_file(filepath):
    """
    Parse one analyzed result txt file.

    Returns a list of dicts — one per Pod/Idle section — with keys:
        arrival_rate, total_pods, idle_timeout,
        blocking_rate_pct  (0-100 scale),
        response_time_s,
        mean_cpu_pct,
        mean_ram_pct
    """
    results = []
    current = None
    # "Pod: 10, Arrival: 0.5, Idle: 0.0"
    section_re = re.compile(
        r'Pod:\s*(\d+),\s*Arrival:\s*([\d.]+),\s*Idle:\s*([\d.]+)'
    )

    def get_mean(pattern, line):
        m = re.search(pattern, line)
        return float(m.group(1)) if m else None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = section_re.search(line)
            if m:
                if current and 'blocking_rate_pct' in current:
                    results.append(current)
                current = {
                    'arrival_rate': float(m.group(2)),
                    'total_pods':   int(m.group(1)),
                    'idle_timeout': float(m.group(3)),
                }
                continue

            if current is None:
                continue

            v = get_mean(r'Blocking rate.*?mean=([\d.]+)%', line)
            if v is not None:
                current['blocking_rate_pct'] = v  # kept in 0-100 range here

            v = get_mean(r'Response time.*?mean=([\d.]+)', line)
            if v is not None:
                current['response_time_s'] = v

            v = get_mean(r'Mean CPU usage.*?mean=([\d.]+)', line)
            if v is not None:
                current['mean_cpu_pct'] = v

            v = get_mean(r'Mean RAM usage.*?mean=([\d.]+)', line)
            if v is not None:
                current['mean_ram_pct'] = v

    if current and 'blocking_rate_pct' in current:
        results.append(current)

    return results


def load_all_experiments(directory):
    """Load all multi_analysis_result_*.txt files from directory."""
    all_rows = []
    pattern = re.compile(r'multi_analysis_result_.*\.txt$')
    if not os.path.isdir(directory):
        return all_rows
    for fname in sorted(os.listdir(directory)):
        if pattern.match(fname):
            fpath = os.path.join(directory, fname)
            print(f"  Reading: {fname}")
            rows = parse_txt_file(fpath)
            print(f"    -> {len(rows)} section(s) found.")
            all_rows.extend(rows)
    return all_rows


# ------------------------------------------------------------------
# CTMC (3D dynamic-pool model)
# ------------------------------------------------------------------

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
            'blocking_probability': metrics['blocking_ratios'][0],  # 0-1
            'latency':              metrics['latency'][0],           # seconds
            'cpu_usage':            metrics['cpu_usage'][0],         # %
            'ram_usage':            metrics['ram_usage'][0],         # %
        }
    except Exception as e:
        print(f"  [ERROR] CTMC failed for lam={arrival_rate}, "
              f"idle={idle_timeout}s, pods={total_pods}: {e}")
        return None


# ------------------------------------------------------------------
# Error metrics
# ------------------------------------------------------------------

def calculate_errors(ctmc_vals, exp_vals, metric_name):
    """Compute MAPE, RMSE, NRMSE, R² and mean signed error."""
    ctmc_vals = np.array(ctmc_vals, dtype=float)
    exp_vals  = np.array(exp_vals,  dtype=float)

    if len(ctmc_vals) == 0:
        return None

    # MAPE: special handling for blocking to avoid div-by-zero when close to 0
    if metric_name == 'blocking_probability':
        if np.mean(exp_vals) < 0.01:
            mape = np.mean(np.abs(ctmc_vals - exp_vals)) * 100
        else:
            mape = np.mean(np.abs((ctmc_vals - exp_vals) / np.maximum(exp_vals, 1e-10))) * 100
    else:
        mape = np.mean(np.abs((ctmc_vals - exp_vals) / np.maximum(np.abs(exp_vals), 1e-10))) * 100

    rmse  = np.sqrt(np.mean((ctmc_vals - exp_vals) ** 2))
    nrmse = (rmse / np.mean(exp_vals) * 100) if np.mean(exp_vals) > 0 else float('inf')

    ss_res = np.sum((ctmc_vals - exp_vals) ** 2)
    ss_tot = np.sum((exp_vals - np.mean(exp_vals)) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return {
        'MAPE':       mape,
        'RMSE':       rmse,
        'NRMSE':      nrmse,
        'R_squared':  r2,
        'mean_error': np.mean(ctmc_vals - exp_vals),
        'mean_ctmc':  np.mean(ctmc_vals),
        'mean_exp':   np.mean(exp_vals),
        'n':          len(ctmc_vals),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    result_dir = os.path.join(_HERE, "result")

    print("=" * 70)
    print("3D CTMC vs Experiment Comparison")
    print("=" * 70)
    print(f"  Service time:       {SERVICE_TIME_S} s  (mu = {MU:.4f} req/s)")
    print(f"  Spawn time:         {SPAWN_TIME_S} s  (spawn_rate = {SPAWN_RATE:.4f} /s)")
    print(f"  CPU warm/active/transit: {CPU_WARM} / {CPU_DEMAND} / {CPU_TRANSIT} %")
    print(f"  RAM warm/active/transit: {RAM_WARM} / {RAM_DEMAND} / {RAM_TRANSIT} %")
    print()

    # --- Load experiments -------------------------------------------------
    print(f"Loading analyzed result files from {result_dir} ...")
    experiments = load_all_experiments(result_dir)
    if not experiments:
        print("No experiment files found. Exiting.")
        return
    print(f"\nTotal data points loaded: {len(experiments)}\n")

    # --- Run CTMC and collect paired results ------------------------------
    ctmc_blocking, exp_blocking = [], []
    ctmc_latency,  exp_latency  = [], []
    ctmc_cpu,      exp_cpu      = [], []
    ctmc_ram,      exp_ram      = [], []

    hdr = (f"{'Arrival':>8} {'Idle(s)':>8} {'Pods':>5}  "
           f"{'Block_CTMC':>11} {'Block_Exp':>10}  "
           f"{'Lat_CTMC':>9} {'Lat_Exp':>8}  "
           f"{'CPU_CTMC':>9} {'CPU_Exp':>8}  "
           f"{'RAM_CTMC':>9} {'RAM_Exp':>8}")
    print(hdr)
    print("-" * len(hdr))

    for exp in experiments:
        lam          = exp['arrival_rate']
        idle_timeout = exp['idle_timeout']
        total_pods   = exp['total_pods']

        ctmc = run_ctmc(lam, idle_timeout, total_pods)
        if ctmc is None:
            continue

        # Experiment blocking is in %, convert to 0-1 for consistent comparison
        exp_block_01 = exp['blocking_rate_pct'] / 100.0
        exp_lat      = exp.get('response_time_s', float('nan'))
        exp_cpu_v    = exp.get('mean_cpu_pct',    float('nan'))
        exp_ram_v    = exp.get('mean_ram_pct',    float('nan'))

        ctmc_blocking.append(ctmc['blocking_probability'])
        exp_blocking.append(exp_block_01)
        ctmc_latency.append(ctmc['latency'])
        exp_latency.append(exp_lat)
        ctmc_cpu.append(ctmc['cpu_usage'])
        exp_cpu.append(exp_cpu_v)
        ctmc_ram.append(ctmc['ram_usage'])
        exp_ram.append(exp_ram_v)

        print(f"{lam:>8.2f} {idle_timeout:>8.1f} {total_pods:>5d}  "
              f"{ctmc['blocking_probability']:>11.4f} {exp_block_01:>10.4f}  "
              f"{ctmc['latency']:>9.4f} {exp_lat:>8.4f}  "
              f"{ctmc['cpu_usage']:>9.3f} {exp_cpu_v:>8.3f}  "
              f"{ctmc['ram_usage']:>9.3f} {exp_ram_v:>8.3f}")

    # --- Error analysis ---------------------------------------------------
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS  (CTMC vs Experiment,  units normalised for each)")
    print("Note: blocking probability is compared in 0-1 scale")
    print("=" * 70)

    comparisons = [
        ('blocking_probability', ctmc_blocking, exp_blocking,
         'Blocking probability (0-1 scale)'),
        ('latency',              ctmc_latency,  exp_latency,
         'Latency / response time  [s]'),
        ('cpu_usage',            ctmc_cpu,      exp_cpu,
         'Mean CPU usage  [%]'),
        ('ram_usage',            ctmc_ram,      exp_ram,
         'Mean RAM usage  [%]'),
    ]

    for metric_name, ctmc_v, exp_v, label in comparisons:
        res = calculate_errors(ctmc_v, exp_v, metric_name)
        if res is None:
            print(f"\n  {label}: no data")
            continue
        print(f"\n  {label}  (n = {res['n']})")
        print(f"    MAPE:         {res['MAPE']:.2f}%")
        print(f"    RMSE:         {res['RMSE']:.6f}")
        print(f"    NRMSE:        {res['NRMSE']:.2f}%")
        print(f"    R-squared:    {res['R_squared']:.4f}")
        print(f"    Mean error:   {res['mean_error']:+.6f}  "
              f"(CTMC avg = {res['mean_ctmc']:.6f},  Exp avg = {res['mean_exp']:.6f})")


if __name__ == '__main__':
    main()
