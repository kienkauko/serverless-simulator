#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script between Markov Model and Simulator — multiprocessing version.

Each test case (Markov + Simulator pair) runs in its own worker process.
All analysis and reporting logic is identical to CTMC_vs_simulator.py.
"""

import numpy as np
import pandas as pd
import math
import copy
import simpy
import sys
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Ensure the project root (3 levels up) is importable so Markov/, variables,
# Server, Request, and fixed_pool resolve regardless of the working directory.
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import Markov model
from Markov.model_2D import MarkovModel

# Import simulator components
from variables import config as base_config
from Server import Server
from Request import Request
from fixed_pool.Container import Container
from fixed_pool.System import System

PEAK_POWER = 150.0

# Batch-means CI configuration (per-scenario steady-state CI on simulator output)
NUM_BATCHES = 20
CI_LEVEL = 0.90  # Confidence level for all CIs
BOOTSTRAP_ITERATIONS = 10000  # Number of bootstrap resamples


def bootstrap_ci(data, stat_func=np.mean, n_boot=BOOTSTRAP_ITERATIONS,
                 ci_level=CI_LEVEL, rng=None):
    """Compute a bootstrap percentile confidence interval.

    Parameters
    ----------
    data : array-like
        1-D sample.
    stat_func : callable
        Statistic to bootstrap (default: np.mean).
    n_boot : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (e.g. 0.90 for 90% CI).
    rng : np.random.Generator or None

    Returns
    -------
    (ci_low, ci_high, point_estimate)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        val = float(data[0])
        return (val, val, val)

    if rng is None:
        rng = np.random.default_rng()

    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([stat_func(data[idx]) for idx in boot_indices])

    alpha = 1 - ci_level
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    point = float(stat_func(data))
    return (ci_low, ci_high, point)


def generate_test_cases(num_cases=500, seed=42):
    """Generate IID test cases by independent uniform sampling over parameter ranges.

    Ranges:
      - num_servers:       Uniform{1, ..., 100}
      - power_scale:       Uniform(0.2, 0.6)
      - warm_percent:      Uniform(0.1, 1.0)
      - mean_service_time: Uniform(1, 50) seconds
      - spawn_time:        Uniform(1, 50) seconds
      - arrival_rate:      Uniform(0.1, 1.0) * total_effective_capacity
      - cpu_warm:          Uniform{0, ..., 50}
      - ram_warm:          Uniform{0, ..., 50}
      - cpu_transit:       Uniform{0, ..., 50}
      - ram_transit:       Uniform{0, ..., 50}
      - cpu_demand:        Uniform{max(5, max(cpu_warm,cpu_transit)+1), ..., 100}
      - ram_demand:        Uniform{max(5, max(ram_warm,ram_transit)+1), ..., 100}

    Active (demand) resources are always strictly greater than both warm and
    transit resources so the scheduling logic is never violated.
    """
    rng = np.random.default_rng(seed)

    cases = []
    while len(cases) < num_cases:
        # Sample warm & transit resources first
        cpu_warm = int(rng.integers(0, 51))
        ram_warm = int(rng.integers(0, 51))
        cpu_transit = int(rng.integers(0, 51))
        ram_transit = int(rng.integers(0, 51))

        # Active must exceed both warm and transit
        cpu_demand_lo = max(5, max(cpu_warm, cpu_transit) + 1)
        ram_demand_lo = max(5, max(ram_warm, ram_transit) + 1)
        if cpu_demand_lo > 100 or ram_demand_lo > 100:
            continue  # reject degenerate draw

        cpu_demand = int(rng.integers(cpu_demand_lo, 101))
        ram_demand = int(rng.integers(ram_demand_lo, 101))

        max_resource = max(cpu_demand, ram_demand)
        containers_per_server = math.floor(100 / max_resource)
        if containers_per_server < 1:
            continue

        num_servers = int(rng.integers(1, 101))
        power_scale = float(rng.uniform(0.2, 0.6))
        warm_percent = float(rng.uniform(0.1, 1.0))
        mean_service_time = float(rng.uniform(1.0, 50.0))
        spawn_time = float(rng.uniform(1.0, 50.0))
        service_rate = 1.0 / mean_service_time

        total_cap = containers_per_server * num_servers
        capacity = total_cap * service_rate

        load_frac = float(rng.uniform(0.1, 1.0))
        arrival_rate = load_frac * capacity

        cases.append({
            'case_id': len(cases),
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'num_servers': num_servers,
            'warm_percent': warm_percent,
            'spawn_time': spawn_time,
            'peak_power': PEAK_POWER,
            'power_scale': power_scale,
            'cpu_warm': cpu_warm,
            'ram_warm': ram_warm,
            'cpu_transit': cpu_transit,
            'ram_transit': ram_transit,
            'cpu_demand': cpu_demand,
            'ram_demand': ram_demand,
        })

    return cases


def convert_to_markov_config(test_case):
    """Convert test case to Markov model configuration"""
    max_resource = max(test_case['cpu_demand'], test_case['ram_demand'])
    total_containers = math.floor(100 / max_resource) * test_case['num_servers']
    queue_warm = int(total_containers * test_case['warm_percent'])
    queue_cold = total_containers - queue_warm

    markov_config = {
        "lam": test_case['arrival_rate'],
        "mu": test_case['service_rate'],
        "spawn_rate": 1.0 / test_case['spawn_time'],
        "queue_warm": queue_warm,
        "queue_cold": queue_cold,
        "serving_time": "exponential",
        "spawn_distribution": "exponential",
        "arrivals": "exponential",
        "ram_warm": test_case['ram_warm'],
        "cpu_warm": test_case['cpu_warm'],
        "ram_demand": test_case['ram_demand'],
        "cpu_demand": test_case['cpu_demand'],
        "cpu_transit": test_case['cpu_transit'],
        "ram_transit": test_case['ram_transit'],
        "peak_power": test_case['peak_power'],
        "power_scale": test_case['power_scale'],
    }

    return markov_config


def convert_to_simulator_config(test_case):
    """Convert test case to simulator configuration."""
    sim_config = copy.deepcopy(base_config)

    sim_config["system"]["num_servers"] = test_case['num_servers']
    sim_config["system"]["warm_percent"] = test_case['warm_percent']
    # Enable per-request records + resource snapshots so compute_batch_means()
    # has data to work with (off by default in System for performance).
    sim_config["system"]["collect_records"] = True
    sim_config["request"]["arrival_rate_mean"] = test_case['arrival_rate']
    sim_config["request"]["arrival_rate_std"] = 0
    sim_config["request"]["service_rate"] = test_case['service_rate']
    sim_config["container"]["spawn_time_mean"] = test_case['spawn_time']
    sim_config["container"]["spawn_time_std"] = 0

    sim_config["request"]["warm_cpu"] = test_case['cpu_warm']
    sim_config["request"]["warm_ram"] = test_case['ram_warm']
    sim_config["request"]["cold_start_cpu"] = test_case['cpu_transit']
    sim_config["request"]["cold_start_ram"] = test_case['ram_transit']
    sim_config["request"]["cpu_demand"] = test_case['cpu_demand']
    sim_config["request"]["ram_demand"] = test_case['ram_demand']

    sim_config["server"]["peak_power"] = test_case['peak_power']
    sim_config["server"]["power_scale"] = test_case['power_scale']

    sim_config["distribution"]["spawn-distribution"] = "exponential"
    sim_config["distribution"]["arrival-distribution"] = "exponential"
    sim_config["distribution"]["service-distribution"] = "exponential"

    return sim_config


def run_markov_model(markov_config):
    """Run the Markov model and extract metrics"""
    try:
        model = MarkovModel(markov_config, verbose=False)
        metrics = model.get_metrics()

        return {
            'blocking_probability': metrics['blocking_ratios'][0],
            'latency': metrics['latency'][0],
            'variance_latency': metrics['variance_latency'][0],
            'p99_latency': metrics['p99_latency'][0],
            'cpu_usage': metrics['cpu_usage'][0],
            'ram_usage': metrics['ram_usage'][0],
            'power_usage': metrics['power_usage'][0],
        }
    except Exception as e:
        print(f"Error in Markov model: {e}")
        return None


def run_simulator(sim_config):
    """Run the simulator and extract metrics with per-scenario 90% CIs from batch means."""
    try:
        env = simpy.Environment()

        system = System(env, sim_config, distribution=sim_config["distribution"],
                       verbose=sim_config["system"]["verbose"])

        for i in range(sim_config["system"]["num_servers"]):
            server = Server(env, f"Server-{i}", sim_config["server"])
            system.add_server(server)

        pre_warm_done = env.process(system.pre_warm())

        def start_request_generator():
            yield pre_warm_done
            env.process(system.request_generator())

        env.process(start_request_generator())
        env.process(system.warmup_reset_process())

        env.run(until=sim_config["system"]["sim_time"])

        batches = system.compute_batch_means(num_batches=NUM_BATCHES)
        boot_rng = np.random.default_rng(42)

        result = {}
        for metric in ['blocking_probability', 'latency', 'variance_latency',
                       'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']:
            vals = np.array(batches[metric], dtype=float)
            n = len(vals)
            if n == 0:
                result[metric] = 0.0
                result[f'{metric}_ci_low'] = 0.0
                result[f'{metric}_ci_high'] = 0.0
                result[f'{metric}_ci_halfwidth'] = 0.0
                result[f'{metric}_n_batches'] = 0
                continue

            ci_low, ci_high, mean_val = bootstrap_ci(
                vals, stat_func=np.mean, ci_level=CI_LEVEL, rng=boot_rng)

            hw = (ci_high - ci_low) / 2.0
            result[metric] = mean_val
            result[f'{metric}_ci_low'] = ci_low
            result[f'{metric}_ci_high'] = ci_high
            result[f'{metric}_ci_halfwidth'] = hw
            result[f'{metric}_n_batches'] = n
        return result

    except Exception as e:
        print(f"Error in simulator: {e}")
        return None


def run_single_case(test_case):
    """Run both Markov model and simulator for a single test case.

    This is the worker function executed in each process.
    Returns (test_case, markov_config, markov_result, sim_result) or None on failure.
    """
    markov_config = convert_to_markov_config(test_case)
    sim_config = convert_to_simulator_config(test_case)

    markov_result = run_markov_model(markov_config)
    sim_result = run_simulator(sim_config)

    return {
        'test_case': test_case,
        'markov_config': markov_config,
        'markov_result': markov_result,
        'sim_result': sim_result,
    }


def calculate_comparison_metrics(markov_results, sim_results):
    """Calculate MAPE, RMSE, and R-squared with bootstrap CIs."""

    metrics = ['blocking_probability', 'latency', 'variance_latency', 'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']
    comparison_results = {}

    for metric in metrics:
        markov_vals = np.array([r[metric] for r in markov_results if r is not None])
        sim_vals = np.array([r[metric] for r in sim_results if r is not None])

        min_len = min(len(markov_vals), len(sim_vals))
        markov_vals = markov_vals[:min_len]
        sim_vals = sim_vals[:min_len]

        n = len(markov_vals)
        if n == 0:
            comparison_results[metric] = {'MAPE': float('inf'), 'RMSE': float('inf'), 'R_squared': 0.0}
            continue

        # Per-case absolute percentage errors
        # For near-zero sim values, use absolute error to avoid division blow-up
        errors = markov_vals - sim_vals
        denom = np.where(sim_vals > 1e-6, sim_vals, 1.0)
        per_case_ape = np.abs(errors / denom) * 100

        mape = float(np.mean(per_case_ape))
        rmse = float(np.sqrt(np.mean((markov_vals - sim_vals) ** 2)))

        ss_res = np.sum((markov_vals - sim_vals) ** 2)
        ss_tot = np.sum((sim_vals - np.mean(sim_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        boot_rng = np.random.default_rng(42)
        if n > 1:
            # Bootstrap CI on MAPE
            ci_mape_low, ci_mape_high, _ = bootstrap_ci(
                per_case_ape, stat_func=np.mean, ci_level=CI_LEVEL, rng=boot_rng)
            # Bootstrap CI on simulator's mean value
            ci_sim_low, ci_sim_high, sim_mean = bootstrap_ci(
                sim_vals, stat_func=np.mean, ci_level=CI_LEVEL, rng=boot_rng)
        else:
            ci_mape_low = ci_mape_high = mape
            sim_mean = float(sim_vals[0])
            ci_sim_low = ci_sim_high = sim_mean

        comparison_results[metric] = {
            'MAPE': mape,
            'RMSE': rmse,
            'R_squared': r_squared,
            'n_samples': n,
            'ci_mape_low': ci_mape_low,
            'ci_mape_high': ci_mape_high,
            'sim_mean': sim_mean,
            'ci_sim_low': ci_sim_low,
            'ci_sim_high': ci_sim_high,
            'markov_mean': float(np.mean(markov_vals)),
        }

    return comparison_results


def main():
    """Main comparison function — multiprocessing version"""
    print("Starting Markov Model vs Simulator Comparison (multi-core)")
    print("=" * 60)

    num_workers = 4
    print(f"Using {num_workers} worker processes")

    # Generate test cases
    test_cases = generate_test_cases(500)
    print(f"Generated {len(test_cases)} test cases")

    # Run all cases in parallel
    print(f"\nDispatching {len(test_cases)} cases to {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_case, test_cases)

    # Collect results (preserve original ordering by case_id)
    markov_results = []
    sim_results = []
    detailed_results = []

    for outcome in results:
        test_case = outcome['test_case']
        markov_config = outcome['markov_config']
        markov_result = outcome['markov_result']
        sim_result = outcome['sim_result']

        if markov_result is not None and sim_result is not None:
            markov_results.append(markov_result)
            sim_results.append(sim_result)

            row = {
                'case_id': test_case['case_id'],
                'arrival_rate': test_case['arrival_rate'],
                'service_rate': test_case['service_rate'],
                'num_servers': test_case['num_servers'],
                'warm_percent': test_case['warm_percent'],
                'spawn_time': test_case['spawn_time'],
                'n_warm': markov_config['queue_warm'],
                'n_cold': markov_config['queue_cold'],
                'cpu_warm': test_case['cpu_warm'],
                'ram_warm': test_case['ram_warm'],
                'cpu_transit': test_case['cpu_transit'],
                'ram_transit': test_case['ram_transit'],
                'cpu_demand': test_case['cpu_demand'],
                'ram_demand': test_case['ram_demand'],
            }
            metric_aliases = {
                'blocking': 'blocking_probability', 'latency': 'latency',
                'variance_latency': 'variance_latency', 'p99_latency': 'p99_latency',
                'cpu': 'cpu_usage', 'ram': 'ram_usage', 'power': 'power_usage',
            }
            for short, full in metric_aliases.items():
                row[f'markov_{short}'] = markov_result[full]
                row[f'sim_{short}'] = sim_result[full]
                row[f'sim_{short}_ci_low'] = sim_result[f'{full}_ci_low']
                row[f'sim_{short}_ci_high'] = sim_result[f'{full}_ci_high']
                row[f'sim_{short}_ci_halfwidth'] = sim_result[f'{full}_ci_halfwidth']
            detailed_results.append(row)

            print(f"  Case {test_case['case_id']:>3d}: "
                  f"block(M={markov_result['blocking_probability']:.4f}, S={sim_result['blocking_probability']:.4f}), "
                  f"latency(M={markov_result['latency']:.4f}, S={sim_result['latency']:.4f})")
        else:
            print(f"  Case {test_case['case_id']:>3d}: FAILED")

    # Calculate comparison metrics
    print(f"\n\nCalculating comparison metrics for {len(markov_results)} successful test cases...")
    comparison_metrics = calculate_comparison_metrics(markov_results, sim_results)

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    ci_pct = int(CI_LEVEL * 100)
    for metric in ['blocking_probability', 'latency', 'variance_latency', 'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        m = comparison_metrics[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}  (n={m['n_samples']}):")
        print(f"    Sim mean:   {m['sim_mean']:.6f}   "
              f"{ci_pct}% CI: [{m['ci_sim_low']:.6f}, {m['ci_sim_high']:.6f}]")
        print(f"    Markov mean:{m['markov_mean']:.6f}")
        print(f"    MAPE:       {m['MAPE']:.2f}%   "
              f"{ci_pct}% CI: [{m['ci_mape_low']:.2f}%, {m['ci_mape_high']:.2f}%]")
        print(f"    RMSE:       {m['RMSE']:.6f}")
        print(f"    R-squared:  {m['R_squared']:.4f}")

    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("comparison_results", exist_ok=True)
    filename = f"comparison_results/markov_vs_sim_{timestamp}.csv"
    pd.DataFrame(detailed_results).to_csv(filename, index=False)
    print(f"\nDetailed results saved to: {filename}")
    print(f"Successful comparisons: {len(markov_results)}/{len(test_cases)} "
          f"({len(markov_results)/len(test_cases)*100:.1f}%), workers: {num_workers}")


if __name__ == "__main__":
    main()
