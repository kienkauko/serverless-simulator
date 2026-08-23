#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare CTMC (Markov/model_2D.py) predictions against real experiment results
stored in txt files inside this directory.

File naming convention expected:
    request_analysis_result_arrival_<rate>_<anything>.txt
    e.g. request_analysis_result_arrival_0.5_2026_04_07_160105.txt

Each txt file contains sections like:
    ══ Pod: 10, Warm: 20% (20 reps) ══
      Blocking rate     :  mean=1.12%  std=2.63%
      Response time  (s):  mean=5.18  std=5.45
      Serving requests  :  mean=2.54  std=0.98
      CPU/req        (%):  mean=5.7804  std=0.2328
      RAM/req        (%):  mean=4.2062  std=1.1594
      Mean CPU usage (%):  mean=14.6969  std=5.7342
      Mean RAM usage (%):  mean=9.8948  std=2.1312
"""

import os
import re
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Allow import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from Markov.model_2D import MarkovModel

# ============================================================
# APPLICATION PROFILE — tune these to match your workload
# ============================================================
SERVICE_TIME_S     = 2.12        # mean service time per request (seconds)
SPAWN_TIME_S       = 11.9        # mean cold-start / spawn time (seconds)
SPAWN_DISTRIBUTION = "exponential"  # "exponential" or "deterministic"

CPU_WARM           = 0.01        # % CPU consumed by one idle warm pod
RAM_WARM           = 2.60        # % RAM consumed by one idle warm pod
CPU_DEMAND         = 5.0         # % CPU consumed when pod is actively serving
RAM_DEMAND         = 2.60        # % RAM consumed when pod is actively serving
CPU_TRANSIT        = 5.35        # % CPU consumed during cold-start transition
RAM_TRANSIT        = 1.51        # % RAM consumed during cold-start transition
PEAK_POWER         = 150.0       # Watts — server peak power
POWER_SCALE        = 0.2         # idle power as fraction of peak power

MU          = 1.0 / SERVICE_TIME_S
SPAWN_RATE  = 1.0 / SPAWN_TIME_S
# ============================================================


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def extract_arrival_rate(filename):
    """Extract float arrival rate from filename (arrival_X.X part)."""
    m = re.search(r'arrival_([\d.]+)', os.path.basename(filename))
    return float(m.group(1)) if m else None


def parse_txt_file(filepath):
    """
    Parse one experiment result txt file.

    Returns a list of dicts — one per Pod/Warm section — with keys:
        arrival_rate, total_pods, warm_percent,
        blocking_rate_pct  (0-100 scale),
        response_time_s,
        serving_requests,
        cpu_per_req_pct,
        ram_per_req_pct,
        mean_cpu_pct,
        mean_ram_pct
    """
    arrival_rate = extract_arrival_rate(filepath)  # None if not in filename

    results = []
    current = None
    # Matches both:
    #   "Pod: 10, Warm: 20%"  (old format — arrival from filename)
    #   "Pod: 10, Arrival: 0.5, Warm: 20%"  (new multi-rate format)
    section_re = re.compile(
        r'Pod:\s*(\d+),\s*(?:Arrival:\s*([\d.]+),\s*)?Warm:\s*([\d.]+)%'
    )

    def get_mean(pattern, line):
        m = re.search(pattern, line)
        return float(m.group(1)) if m else None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = section_re.search(line)
            if m:
                # Save completed previous section
                if current and 'blocking_rate_pct' in current:
                    results.append(current)
                # Arrival rate: prefer value from section header, fall back to filename
                section_arrival = float(m.group(2)) if m.group(2) else arrival_rate
                if section_arrival is None:
                    print(f"  [WARN] No arrival rate for section in {filepath}, skipping.")
                    current = None
                    continue
                current = {
                    'arrival_rate': section_arrival,
                    'total_pods':   int(m.group(1)),
                    'warm_percent': float(m.group(3)) / 100.0,
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

            v = get_mean(r'Serving requests.*?mean=([\d.]+)', line)
            if v is not None:
                current['serving_requests'] = v

            # CPU/req (%) — match only the "%" variant, not the MB one
            v = get_mean(r'CPU/req\s+\(%\).*?mean=([\d.]+)', line)
            if v is not None:
                current['cpu_per_req_pct'] = v

            # RAM/req (%) — same, not MB
            v = get_mean(r'RAM/req\s+\(%\).*?mean=([\d.]+)', line)
            if v is not None:
                current['ram_per_req_pct'] = v

            v = get_mean(r'Mean CPU usage.*?mean=([\d.]+)', line)
            if v is not None:
                current['mean_cpu_pct'] = v

            v = get_mean(r'Mean RAM usage.*?mean=([\d.]+)', line)
            if v is not None:
                current['mean_ram_pct'] = v

            # Processing time P99 and variance
            m_p99 = re.search(r'Processing time.*?p99=([\d.]+)', line)
            if m_p99 is not None:
                current['processing_p99_s'] = float(m_p99.group(1))
            m_var = re.search(r'Processing time.*?var=([\d.]+)', line)
            if m_var is not None:
                current['processing_var_s2'] = float(m_var.group(1))

    # Save the last section
    if current and 'blocking_rate_pct' in current:
        results.append(current)

    return results


def load_all_experiments(directory):
    """Load all matching txt files from directory, return combined list."""
    all_rows = []
    patterns = [
        re.compile(r'request_analysis_result_arrival_[\d.]+.*\.txt$'),
        re.compile(r'multi_analysis_result_.*\.txt$'),
    ]
    for fname in sorted(os.listdir(directory)):
        if any(p.match(fname) for p in patterns):
            fpath = os.path.join(directory, fname)
            print(f"  Reading: {fname}")
            rows = parse_txt_file(fpath)
            print(f"    -> {len(rows)} section(s) found.")
            all_rows.extend(rows)
    return all_rows


# ------------------------------------------------------------------
# CTMC
# ------------------------------------------------------------------

def run_ctmc(arrival_rate, warm_percent, total_pods):
    """
    Run CTMC model for the given configuration.

    queue_warm = floor(total_pods * warm_percent)
    queue_cold = total_pods - queue_warm
    """
    queue_warm = int(total_pods * warm_percent)
    queue_cold = total_pods - queue_warm

    config = {
        "lam":               arrival_rate,
        "mu":                MU,
        "spawn_rate":        SPAWN_RATE,
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
            'blocking_probability': metrics['blocking_ratios'][0],  # 0-1
            'latency':              metrics['latency'][0],           # seconds
            'cpu_usage':            metrics['cpu_usage'][0],         # %
            'ram_usage':            metrics['ram_usage'][0],         # %
            'p99_latency':          metrics['p99_latency'][0],       # seconds
            'variance_latency':     metrics['variance_latency'][0],  # seconds^2
        }
    except Exception as e:
        print(f"  [ERROR] CTMC failed for lam={arrival_rate}, "
              f"warm={warm_percent*100:.0f}%, pods={total_pods}: {e}")
        return None


# ------------------------------------------------------------------
# Error metrics  (same style as CTMC_vs_simulator_traces.py)
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
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("CTMC vs Experiment Comparison")
    print("=" * 70)
    print(f"  Service time:       {SERVICE_TIME_S} s  (mu = {MU:.4f} req/s)")
    print(f"  Spawn time:         {SPAWN_TIME_S} s  (spawn_rate = {SPAWN_RATE:.4f} /s)")
    print(f"  Spawn distribution: {SPAWN_DISTRIBUTION}")
    print(f"  CPU warm / demand:  {CPU_WARM} % / {CPU_DEMAND} %")
    print(f"  RAM warm / demand:  {RAM_WARM} % / {RAM_DEMAND} %")
    print()

    # --- Load experiments -------------------------------------------------
    print("Loading experiment files...")
    experiments = load_all_experiments(script_dir)
    if not experiments:
        print("No experiment files found. Exiting.")
        return
    print(f"\nTotal data points loaded: {len(experiments)}\n")

    # --- Run CTMC and collect paired results ------------------------------
    ctmc_blocking, exp_blocking = [], []
    ctmc_latency,  exp_latency  = [], []
    ctmc_cpu,      exp_cpu      = [], []
    ctmc_ram,      exp_ram      = [], []
    ctmc_p99,      exp_p99      = [], []
    ctmc_var,      exp_var      = [], []

    hdr = (f"{'Arrival':>8} {'Warm%':>6} {'Pods':>5}  "
           f"{'Block_CTMC':>11} {'Block_Exp':>10}  "
           f"{'Lat_CTMC':>9} {'Lat_Exp':>8}  "
           f"{'P99_CTMC':>9} {'P99_Exp':>8}  "
           f"{'Var_CTMC':>9} {'Var_Exp':>8}  "
           f"{'CPU_CTMC':>9} {'CPU_Exp':>8}  "
           f"{'RAM_CTMC':>9} {'RAM_Exp':>8}")
    print(hdr)
    print("-" * len(hdr))

    for exp in experiments:
        lam        = exp['arrival_rate']
        warm_pct   = exp['warm_percent']
        total_pods = exp['total_pods']

        ctmc = run_ctmc(lam, warm_pct, total_pods)
        if ctmc is None:
            continue

        # Experiment blocking is in %, convert to 0-1 for consistent comparison
        exp_block_01 = exp['blocking_rate_pct'] / 100.0
        exp_lat      = exp.get('response_time_s',    float('nan'))
        exp_cpu_v    = exp.get('mean_cpu_pct',        float('nan'))
        exp_ram_v    = exp.get('mean_ram_pct',        float('nan'))
        exp_p99_v    = exp.get('processing_p99_s',    float('nan'))
        exp_var_v    = exp.get('processing_var_s2',   float('nan'))

        ctmc_blocking.append(ctmc['blocking_probability'])
        exp_blocking.append(exp_block_01)
        ctmc_latency.append(ctmc['latency'])
        exp_latency.append(exp_lat)
        ctmc_cpu.append(ctmc['cpu_usage'])
        exp_cpu.append(exp_cpu_v)
        ctmc_ram.append(ctmc['ram_usage'])
        exp_ram.append(exp_ram_v)

        if not np.isnan(exp_p99_v):
            ctmc_p99.append(ctmc['p99_latency'])
            exp_p99.append(exp_p99_v)
        if not np.isnan(exp_var_v):
            ctmc_var.append(ctmc['variance_latency'])
            exp_var.append(exp_var_v)

        print(f"{lam:>8.2f} {warm_pct*100:>5.0f}% {total_pods:>5d}  "
              f"{ctmc['blocking_probability']:>11.4f} {exp_block_01:>10.4f}  "
              f"{ctmc['latency']:>9.4f} {exp_lat:>8.4f}  "
              f"{ctmc['p99_latency']:>9.4f} {exp_p99_v:>8.4f}  "
              f"{ctmc['variance_latency']:>9.4f} {exp_var_v:>8.4f}  "
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
        ('p99_latency',          ctmc_p99,      exp_p99,
         'P99 processing time  [s]'),
        ('variance_latency',     ctmc_var,      exp_var,
         'Variance of processing time  [s^2]'),
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
