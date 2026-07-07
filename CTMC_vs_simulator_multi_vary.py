#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script between Markov Model and Simulator (multiprocessing version).

VARYING-RESOURCE variant
=========================
Unlike ``CTMC_vs_simulator_multi.py``, this version does NOT fix the per-function
resource consumption (warm / active / transit usage). Instead every scenario draws
*all* parameters uniformly at random from the validation ranges below. There are no
hand-crafted extreme cases.

Validation ranges (each scenario samples uniformly and independently):

    Number of devices (n_p)          : [10 - 100]
    Device base power (p_base)       : [20% - 60%] of p_max      -> power_scale
    Device peak power (p_max)        : 150 W                     (fixed)
    Arrival rate (lambda)            : [10% - 100%] of lambda_max (offered load)
    Num. warm functions (n_w)        : [10% - 100%] of n
    Mean service time (E[B_w])       : [1 - 30] s                -> mu = 1/E[B_w]
    Warm usage (r_w)                 : [0% - 50%]  of r_cap
    Active usage (r_a)               : [1% - 100%] of r_cap
    Transit usage (r_t)              : [0% - 50%]  of r_cap
    Mean null-to-warm time (E[B_c])  : [0.1 - 30] s              -> spawn_time

Notes:
- ``r`` represents BOTH RAM and CPU resources. CPU and RAM usages are drawn
  independently per resource (so cpu_warm != ram_warm in general).
- ``r_a`` is only valid when it is >= r_w AND >= r_t (enforced per resource by
  rejection sampling).
- r_cap = 100 (%). Containers per device = floor(r_cap / max(cpu_active, ram_active)).
- To keep the CTMC state space tractable, scenarios whose total container count
  (containers_per_server * n_p) exceeds MAX_TOTAL_CONTAINERS are resampled.
- Fully-warm scenarios (n_cold = 0) are allowed.

This variant parallelizes the per-scenario work (Markov model + N simulator
replications) across multiple worker processes to accelerate total runtime.
"""

import numpy as np
import pandas as pd
import math
import copy
import random
import simpy
import sys
import os
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Import Markov model
from Markov.model_2D import MarkovModel

# Import simulator components
from variables import config as base_config
from Server import Server
from Request import Request
from Container import Container
from System import System

# --- Sampling constants -----------------------------------------------------
R_CAP = 100.0                 # Resource cap (%) for both CPU and RAM
PEAK_POWER = 150.0            # Device peak power p_max (W)
MAX_TOTAL_CONTAINERS = 500    # Cap on containers_per_server * n_p (CTMC tractability)


def sample_resource_triplet(rng):
    """Sample (warm, transit, active) usage for ONE resource, in % of r_cap.

    Ranges: warm in [0, 50], transit in [0, 50], active in [1, 100].
    Enforces active >= warm and active >= transit via rejection sampling
    (so the marginals remain uniform within the valid region).
    """
    while True:
        r_w = rng.uniform(0.0, 50.0)
        r_t = rng.uniform(0.0, 50.0)
        r_a = rng.uniform(1.0, 100.0)
        if r_a >= r_w and r_a >= r_t:
            return r_w, r_t, r_a


def generate_test_cases(num_cases=100, seed=42):
    """Generate ``num_cases`` scenarios, each drawn uniformly at random from the
    validation ranges. Scenarios exceeding MAX_TOTAL_CONTAINERS are resampled so
    that exactly ``num_cases`` valid scenarios are returned.
    """
    rng = random.Random(seed)
    cases = []

    for case_id in range(num_cases):
        while True:
            num_servers = rng.randint(10, 100)                  # n_p
            power_scale = rng.uniform(0.2, 0.6)                 # p_base as fraction of p_max
            service_time = rng.uniform(1.0, 30.0)              # E[B_w] (s)
            service_rate = 1.0 / service_time                  # mu
            spawn_time = rng.uniform(0.1, 30.0)                # E[B_c] (s)
            warm_percent = rng.uniform(0.1, 1.0)               # n_w as fraction of n

            # Per-resource usage triplets (independent CPU / RAM draws)
            cpu_w, cpu_t, cpu_a = sample_resource_triplet(rng)
            ram_w, ram_t, ram_a = sample_resource_triplet(rng)

            # Container packing determined by peak (active) usage
            max_active = max(cpu_a, ram_a)
            containers_per_server = math.floor(R_CAP / max_active)
            if containers_per_server < 1:
                continue
            total_containers = containers_per_server * num_servers
            if total_containers > MAX_TOTAL_CONTAINERS:
                continue

            n_warm = int(total_containers * warm_percent)
            if n_warm < 1:
                n_warm = 1
            n_cold = total_containers - n_warm  # may be 0 (fully-warm allowed)

            # Offered-load capacity (lambda_max) and arrival rate
            warm_capacity = n_warm * service_rate
            cold_effective_rate = 1.0 / (spawn_time + 1.0 / service_rate)
            lambda_max = warm_capacity + n_cold * cold_effective_rate
            load_frac = rng.uniform(0.1, 1.0)                  # lambda as fraction of lambda_max
            arrival_rate = load_frac * lambda_max
            if arrival_rate <= 0:
                continue
            break

        cases.append({
            'case_id': case_id,
            'arrival_rate': round(arrival_rate, 4),
            'service_rate': service_rate,
            'service_time': service_time,
            'num_servers': num_servers,
            'warm_percent': warm_percent,
            'spawn_time': spawn_time,
            'power_scale': power_scale,
            # Per-resource usages (% of r_cap)
            'cpu_warm': cpu_w, 'ram_warm': ram_w,
            'cpu_transit': cpu_t, 'ram_transit': ram_t,
            'cpu_active': cpu_a, 'ram_active': ram_a,
            # Derived sizing
            'containers_per_server': containers_per_server,
            'total_containers': total_containers,
            'n_warm': n_warm,
            'n_cold': n_cold,
            'load_frac': load_frac,
        })

    return cases


def convert_to_markov_config(test_case):
    """Convert test case to Markov model configuration"""

    # Queue sizes based on simulator packing logic (peak/active usage packs)
    max_active = max(test_case['cpu_active'], test_case['ram_active'])
    total_containers = math.floor(R_CAP / max_active) * test_case['num_servers']
    queue_warm = int(total_containers * test_case['warm_percent'])
    if queue_warm < 1:
        queue_warm = 1
    queue_cold = total_containers - queue_warm

    markov_config = {
        "lam": test_case['arrival_rate'],
        "mu": test_case['service_rate'],
        "spawn_rate": 1.0 / test_case['spawn_time'],
        "queue_warm": queue_warm,
        "queue_cold": queue_cold,
        "serving_time": "exponential",
        "arrivals": "exponential",
        # Warm-state usage
        "ram_warm": test_case['ram_warm'],
        "cpu_warm": test_case['cpu_warm'],
        # Transit (cold-start) usage
        "ram_cold": test_case['ram_transit'],
        "cpu_cold": test_case['cpu_transit'],
        # Active-state (demand) usage
        "ram_demand": test_case['ram_active'],
        "cpu_demand": test_case['cpu_active'],
        "peak_power": PEAK_POWER,
        "power_scale": test_case['power_scale'],
    }

    return markov_config


def convert_to_simulator_config(test_case):
    """Convert test case to simulator configuration.

    Deep-copies the full config from variables.py and only overrides
    the parameters that are varied in the test case.
    """
    sim_config = copy.deepcopy(base_config)

    # Override only the test-case parameters
    sim_config["system"]["num_servers"] = test_case['num_servers']
    sim_config["system"]["warm_percent"] = test_case['warm_percent']
    sim_config["request"]["arrival_rate_mean"] = test_case['arrival_rate']
    sim_config["request"]["arrival_rate_std"] = 0
    sim_config["request"]["service_rate"] = test_case['service_rate']
    sim_config["container"]["spawn_time_mean"] = test_case['spawn_time']
    sim_config["container"]["spawn_time_std"] = 0

    # Per-resource usages (independent CPU / RAM)
    sim_config["request"]["warm_cpu"] = test_case['cpu_warm']
    sim_config["request"]["warm_ram"] = test_case['ram_warm']
    sim_config["request"]["cold_start_cpu"] = test_case['cpu_transit']
    sim_config["request"]["cold_start_ram"] = test_case['ram_transit']
    sim_config["request"]["cpu_demand"] = test_case['cpu_active']
    sim_config["request"]["ram_demand"] = test_case['ram_active']

    # Power parameters
    sim_config["server"]["peak_power"] = PEAK_POWER
    sim_config["server"]["power_scale"] = test_case['power_scale']

    # Ensure distributions match Markov model assumptions (M/M type)
    sim_config["distribution"]["spawn-distribution"] = "deterministic"  # Deterministic spawn time for simulator
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
    """Run the simulator and extract metrics"""
    try:
        # Create simulation environment
        env = simpy.Environment()

        # Create system
        system = System(env, sim_config, distribution=sim_config["distribution"],
                       verbose=sim_config["system"]["verbose"])

        # Add servers
        for i in range(sim_config["system"]["num_servers"]):
            server = Server(env, f"Server-{i}", sim_config["server"])
            system.add_server(server)

        # Start pre-warming process
        pre_warm_done = env.process(system.pre_warm())

        # Start request generation after pre-warming
        def start_request_generator():
            yield pre_warm_done
            env.process(system.request_generator())

        env.process(start_request_generator())
        env.process(system.resource_monitor_process())

        # Run simulation
        env.run(until=sim_config["system"]["sim_time"])

        # Calculate metrics
        blocking_probability = 0.0
        if system.request_stats['generated'] > 0:
            blocking_probability = system.request_stats['blocked_no_server_capacity'] / system.request_stats['generated']

        avg_latency = 0.0
        variance_latency = 0.0
        p99_latency = 0.0
        if system.latency_stats['count'] > 0:
            n = system.latency_stats['count']
            avg_latency = system.latency_stats['total_latency'] / n
            e_br2 = system.latency_stats['total_latency_sq'] / n
            variance_latency = max(e_br2 - avg_latency ** 2, 0.0)
            all_lat = system.latency_stats['all_latencies']
            if len(all_lat) > 0:
                p99_latency = float(np.percentile(all_lat, 99))
        mean_cpu_usage = system.get_mean_cpu_usage()
        mean_ram_usage = system.get_mean_ram_usage()
        mean_power_usage = system.get_mean_power_usage()

        return {
            'blocking_probability': blocking_probability,
            'latency': avg_latency,
            'variance_latency': variance_latency,
            'p99_latency': p99_latency,
            'cpu_usage': mean_cpu_usage,
            'ram_usage': mean_ram_usage,
            'power_usage': mean_power_usage
        }

    except Exception as e:
        print(f"Error in simulator: {e}")
        return None

# Metrics tracked throughout the comparison
METRICS_LIST = ['blocking_probability', 'latency', 'variance_latency',
                'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']

def process_test_case(test_case, num_sim_replications):
    """Run one scenario (Markov once + N simulator replications) end-to-end.

    This is the unit of work dispatched to each worker process. It is a
    top-level function (required for multiprocessing on Windows 'spawn')
    and returns a fully self-contained result dict, or None on failure.
    """
    case_id = test_case['case_id']

    # Convert to respective configurations
    markov_config = convert_to_markov_config(test_case)
    sim_config = convert_to_simulator_config(test_case)

    # Run Markov model (deterministic — only once)
    markov_result = run_markov_model(markov_config)

    # Run simulator num_sim_replications times
    sim_replication_results = []
    for _ in range(num_sim_replications):
        res = run_simulator(sim_config)
        if res is not None:
            sim_replication_results.append(res)

    if markov_result is None or len(sim_replication_results) < 2:
        print(f"[case {case_id}] ERROR: Markov failed or insufficient sim replications!")
        return None

    # Mean over replications for each metric (the single sim value per scenario;
    # across-scenario uncertainty is quantified later via bootstrap CI)
    n_reps = len(sim_replication_results)
    sim_mean = {}
    for metric in METRICS_LIST:
        vals = np.array([r[metric] for r in sim_replication_results])
        sim_mean[metric] = np.mean(vals)

    # Build detailed results row
    row = {
        'case_id': test_case['case_id'],
        'arrival_rate': test_case['arrival_rate'],
        'service_rate': test_case['service_rate'],
        'service_time': test_case['service_time'],
        'num_servers': test_case['num_servers'],
        'warm_percent': test_case['warm_percent'],
        'spawn_time': test_case['spawn_time'],
        'power_scale': test_case['power_scale'],
        'cpu_warm': test_case['cpu_warm'],
        'ram_warm': test_case['ram_warm'],
        'cpu_transit': test_case['cpu_transit'],
        'ram_transit': test_case['ram_transit'],
        'cpu_active': test_case['cpu_active'],
        'ram_active': test_case['ram_active'],
        'containers_per_server': test_case['containers_per_server'],
        'total_containers': test_case['total_containers'],
        'n_warm': markov_config['queue_warm'],
        'n_cold': markov_config['queue_cold'],
        'n_reps': n_reps,
    }
    for metric in METRICS_LIST:
        row[f'markov_{metric}'] = markov_result[metric]
        row[f'sim_mean_{metric}'] = sim_mean[metric]

    # Compact per-case log (prints from workers may interleave)
    print(f"[case {case_id}] done: "
          f"λ={test_case['arrival_rate']:.1f}, μ={test_case['service_rate']:.3f}, "
          f"servers={test_case['num_servers']}, warm%={test_case['warm_percent']:.2f}, "
          f"spawn={test_case['spawn_time']:.1f} | "
          f"Markov block={markov_result['blocking_probability']:.4f} lat={markov_result['latency']:.4f} | "
          f"Sim block={sim_mean['blocking_probability']:.4f} lat={sim_mean['latency']:.4f}")

    return {
        'case_id': case_id,
        'markov_result': markov_result,
        'sim_mean': sim_mean,
        'row': row,
    }

def bootstrap_ci_on_sim_mean(markov_results, sim_results, n_boot=10000,
                             ci_level=0.95, seed=12345):
    """Bootstrap CI on the simulator's ACROSS-SCENARIO mean for each metric.

    This validates the (deterministic) Markov model the way the paper reports:
    for every metric we build a ``ci_level`` bootstrap confidence interval on
    the simulator's mean over all scenarios (resampling scenarios with
    replacement), then check whether the Markov model's across-scenario mean
    prediction falls inside that interval.

    Returns a dict keyed by metric with sim_mean, ci_lower, ci_upper,
    markov_mean and a boolean ``within``.
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci_level
    results = {}

    for metric in METRICS_LIST:
        markov_vals = np.array([r[metric] for r in markov_results if r is not None])
        sim_vals = np.array([r[metric] for r in sim_results if r is not None])

        n = min(len(markov_vals), len(sim_vals))
        markov_vals = markov_vals[:n]
        sim_vals = sim_vals[:n]
        if n == 0:
            continue

        # Resample scenarios with replacement and recompute the mean each time
        boot_idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = sim_vals[boot_idx].mean(axis=1)

        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        sim_mean = float(sim_vals.mean())
        markov_mean = float(markov_vals.mean())

        results[metric] = {
            'sim_mean': sim_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'markov_mean': markov_mean,
            'within': ci_lower <= markov_mean <= ci_upper,
        }

    return results


def calculate_comparison_metrics(markov_results, sim_results):
    """Calculate MAPE, RMSE, and R-squared for comparison"""

    metrics = ['blocking_probability', 'latency', 'variance_latency', 'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']
    comparison_results = {}

    for metric in metrics:
        markov_vals = np.array([r[metric] for r in markov_results if r is not None])
        sim_vals = np.array([r[metric] for r in sim_results if r is not None])

        # Ensure we have the same number of valid results
        min_len = min(len(markov_vals), len(sim_vals))
        markov_vals = markov_vals[:min_len]
        sim_vals = sim_vals[:min_len]

        if len(markov_vals) == 0:
            comparison_results[metric] = {'MAPE': float('inf'), 'RMSE': float('inf'), 'R_squared': 0.0}
            continue

        # Calculate MAPE (Mean Absolute Percentage Error) - Markov vs Simulator reference
        # Special handling for very small values (like blocking probability)
        if metric == 'blocking_probability':
            # For blocking probability, use absolute error if values are very small
            threshold = 0.01  # 1% blocking probability threshold
            if np.mean(sim_vals) < threshold:
                # Use Mean Absolute Error instead of MAPE for very small values
                mape = np.mean(np.abs(markov_vals - sim_vals)) * 100  # Convert to percentage points
            else:
                mape = np.mean(np.abs((markov_vals - sim_vals) / np.maximum(sim_vals, 1e-10))) * 100
        else:
            mape = np.mean(np.abs((markov_vals - sim_vals) / np.maximum(sim_vals, 1e-10))) * 100

        # Calculate RMSE (Root Mean Square Error) - Markov vs Simulator reference
        rmse = np.sqrt(np.mean((markov_vals - sim_vals) ** 2))

        # Calculate Normalized RMSE (NRMSE) as percentage of mean
        nrmse = (rmse / np.mean(sim_vals)) * 100 if np.mean(sim_vals) > 0 else float('inf')

        # Calculate R-squared - Using Simulator as reference
        ss_res = np.sum((markov_vals - sim_vals) ** 2)
        ss_tot = np.sum((sim_vals - np.mean(sim_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Signed mean error (bias): positive means CTMC overestimates
        mean_error = np.mean(markov_vals - sim_vals)

        comparison_results[metric] = {
            'MAPE': mape,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'R_squared': r_squared,
            'mean_error': mean_error,
            'mean_markov': np.mean(markov_vals),
            'mean_sim': np.mean(sim_vals),
        }

    return comparison_results

def main():
    """Main comparison function"""
    NUM_SIM_REPLICATIONS = 20
    CI_LEVEL = 0.95

    # Number of worker processes. Override via CLI arg, else use all cores.
    if len(sys.argv) > 1:
        num_workers = int(sys.argv[1])
    else:
        num_workers = cpu_count()

    print("Starting Markov Model vs Simulator Comparison (multiprocessing, varying resources)")
    print(f"  Simulation replications per scenario: {NUM_SIM_REPLICATIONS}")
    print(f"  Confidence level: {CI_LEVEL*100:.0f}%")
    print(f"  Worker processes: {num_workers}")
    print("=" * 60)

    # Generate test cases
    test_cases = generate_test_cases(500)
    print(f"Generated {len(test_cases)} test cases")

    metrics_list = METRICS_LIST

    # Dispatch each test case to a worker process
    worker = partial(process_test_case,
                     num_sim_replications=NUM_SIM_REPLICATIONS)

    raw_results = []
    with Pool(processes=num_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(worker, test_cases)):
            print(f"Progress: {idx + 1}/{len(test_cases)} test cases completed")
            if result is not None:
                raw_results.append(result)

    # Sort back into deterministic (case_id) order for reproducible reporting
    raw_results.sort(key=lambda r: r['case_id'])

    # Unpack aggregated results
    markov_results = [r['markov_result'] for r in raw_results]
    sim_results = [r['sim_mean'] for r in raw_results]  # stores the mean across replications
    detailed_results = [r['row'] for r in raw_results]

    # Calculate comparison metrics (using sim means as the single sim value)
    print(f"\n\nCalculating comparison metrics for {len(markov_results)} successful test cases...")
    comparison_metrics = calculate_comparison_metrics(markov_results, sim_results)

    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    for metric in metrics_list:
        m = comparison_metrics[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"    MAPE:       {m['MAPE']:.2f}%")
        print(f"    RMSE:       {m['RMSE']:.6f}")
        print(f"    NRMSE:      {m['NRMSE']:.2f}%")
        print(f"    R-squared:  {m['R_squared']:.4f}")
        print(f"    Mean Error: {m['mean_error']:+.6f}  "
              f"(CTMC avg={m['mean_markov']:.6f}, Sim avg={m['mean_sim']:.6f})")

    # =========================================================================
    # Aggregate bootstrap CI on the simulator's across-scenario mean
    # (Markov prediction should fall inside — matches the paper's validation)
    # =========================================================================
    N_BOOT = 10000
    boot_ci = bootstrap_ci_on_sim_mean(markov_results, sim_results,
                                       n_boot=N_BOOT, ci_level=CI_LEVEL)
    print(f"\n{'=' * 70}")
    print(f"  BOOTSTRAP {CI_LEVEL*100:.0f}% CI ON SIMULATOR MEAN (across all scenarios)")
    print(f"  Bootstrap resamples: {N_BOOT} | Markov mean should fall inside CI")
    print(f"{'=' * 70}")
    for metric in metrics_list:
        if metric not in boot_ci:
            continue
        b = boot_ci[metric]
        flag = "OK  (within CI)" if b['within'] else "OUT (outside CI)"
        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"    Sim mean:   {b['sim_mean']:.6f}  "
              f"CI: [{b['ci_lower']:.6f}, {b['ci_upper']:.6f}]")
        print(f"    Markov:     {b['markov_mean']:.6f}   -> {flag}")

    # -------------------------------------------------------------------------
    # Bias & Error Distribution Analysis (CTMC vs Sim mean)
    # -------------------------------------------------------------------------
    if len(detailed_results) > 0:
        df = pd.DataFrame(detailed_results)
        print(f"\n{'=' * 70}")
        print(f"  BIAS & ERROR DISTRIBUTION ANALYSIS (CTMC vs Simulator mean)")
        print(f"{'=' * 70}")

        for metric_short, metric_label in [
            ('blocking_probability', 'Blocking Prob.'), ('latency', 'Latency'),
            ('variance_latency', 'Variance of Latency'), ('p99_latency', 'P99 Latency'),
            ('cpu_usage', 'CPU Usage'), ('ram_usage', 'RAM Usage'), ('power_usage', 'Power Usage'),
        ]:
            model_col = f'markov_{metric_short}'
            sim_col = f'sim_mean_{metric_short}'
            if model_col not in df.columns or sim_col not in df.columns:
                continue

            errors = df[model_col] - df[sim_col]
            abs_errors = errors.abs()

            print(f"\n  {metric_label}:")
            print(f"    Mean Signed Error (bias):  {errors.mean():+.6f}")
            print(f"    Std of Error:              {errors.std():.6f}")
            print(f"    Median Signed Error:       {errors.median():+.6f}")
            print(f"    Mean |Error|:              {abs_errors.mean():.6f}")
            print(f"    Error Percentiles:")
            for p in [5, 25, 50, 75, 95]:
                print(f"      P{p:02d}: {errors.quantile(p/100):+.6f}")

            n_over = (errors > 1e-9).sum()
            n_under = (errors < -1e-9).sum()
            n_match = len(errors) - n_over - n_under
            print(f"    Overestimates:  {n_over}/{len(errors)} ({100*n_over/len(errors):.1f}%)")
            print(f"    Underestimates: {n_under}/{len(errors)} ({100*n_under/len(errors):.1f}%)")
            print(f"    Near-exact:     {n_match}/{len(errors)} ({100*n_match/len(errors):.1f}%)")

    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results/markov_vs_sim_vary_{timestamp}.csv"

    os.makedirs("comparison_results", exist_ok=True)

    if len(detailed_results) > 0:
        df_out = pd.DataFrame(detailed_results)
        df_out.to_csv(filename, index=False)
        print(f"\nDetailed results saved to: {filename}")

    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful comparisons: {len(markov_results)}")
    print(f"Success rate: {len(markov_results)/len(test_cases)*100:.1f}%")
    print(f"Simulation replications per scenario: {NUM_SIM_REPLICATIONS}")
    print(f"Confidence level: {CI_LEVEL*100:.0f}%")
    print(f"Worker processes: {num_workers}")

if __name__ == "__main__":
    main()
