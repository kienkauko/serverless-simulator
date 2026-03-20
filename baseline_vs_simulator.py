#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script: Cascaded Erlang B Baseline vs Simulator
Runs both systems with the same input parameters and compares their results.

This highlights the inaccuracy of the decomposed Erlang B approach
(which assumes independent pools and Poisson overflow) by comparing
it directly against the discrete-event simulator.
"""

import numpy as np
import pandas as pd
import math
import copy
import simpy
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import simulator components
from Server import Server
from Request import Request
from Container import Container
from System import System
from variables import config as base_config

# Import the cascaded Erlang B baseline from CTMC_vs_baseline
from CTMC_vs_baseline import CascadedErlangBModel

cpu_warm = 1
ram_warm = 30
cpu_demand = 50
ram_demand = 40

# =============================================================================
# Test Case Generation & Config Conversion
# =============================================================================

def generate_test_cases(num_cases=500):
    """Generate test cases covering a wide range of scenarios, including
    extreme cases that stress the independence assumption.

    Covers:
    - Very small pools (1-2 servers) where coupling is strongest
    - Moderate pools (3-10 servers) in the interesting regime
    - Larger pools (15-20 servers) as a baseline
    - Long cold start times (up to 25s) that amplify coupling effects
    - Various load levels from light to overloaded
    """
    all_cases = []

    max_resource = max(cpu_demand, ram_demand)
    containers_per_server = math.floor(100 / max_resource)  # = 2

    # Expanded parameter grids
    num_servers_list = [1, 2, 3, 4, 5, 8, 10, 15, 20]
    warm_percents = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    spawn_times = [1, 2, 4, 6, 8, 12, 16, 20, 25]
    service_rates = [0.25, 0.5, 1.0, 2.0, 4.0]
    load_fractions = [0.5, 0.7, 0.85, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]

    for num_servers in num_servers_list:
        total_cap = containers_per_server * num_servers
        for service_rate in service_rates:
            for warm_percent in warm_percents:
                n_warm = int(total_cap * warm_percent)
                n_cold = total_cap - n_warm

                if n_warm < 1 or n_cold < 1:
                    continue

                warm_capacity = n_warm * service_rate

                for spawn_time in spawn_times:
                    cold_effective_rate = 1.0 / (spawn_time + 1.0 / service_rate)
                    total_effective_capacity = warm_capacity + n_cold * cold_effective_rate

                    for frac in load_fractions:
                        arrival_rate = frac * total_effective_capacity
                        if arrival_rate < 0.1:
                            continue

                        rho_warm = arrival_rate / service_rate
                        if rho_warm < n_warm * 0.3:
                            continue

                        all_cases.append({
                            'arrival_rate': round(arrival_rate, 2),
                            'service_rate': service_rate,
                            'num_servers': num_servers,
                            'warm_percent': warm_percent,
                            'spawn_time': spawn_time,
                        })

    # Extreme coupling cases: hand-crafted for maximum stress
    extreme_cases = [
        # Tiny systems (1-2 servers) — strongest coupling
        {'arrival_rate': 0.8, 'service_rate': 1.0, 'num_servers': 1,
         'warm_percent': 0.5, 'spawn_time': 10},
        {'arrival_rate': 1.2, 'service_rate': 1.0, 'num_servers': 1,
         'warm_percent': 0.5, 'spawn_time': 15},
        {'arrival_rate': 1.5, 'service_rate': 1.0, 'num_servers': 1,
         'warm_percent': 0.5, 'spawn_time': 20},
        {'arrival_rate': 0.5, 'service_rate': 0.5, 'num_servers': 1,
         'warm_percent': 0.5, 'spawn_time': 25},
        {'arrival_rate': 1.0, 'service_rate': 2.0, 'num_servers': 1,
         'warm_percent': 0.5, 'spawn_time': 10},
        {'arrival_rate': 2.0, 'service_rate': 1.0, 'num_servers': 2,
         'warm_percent': 0.25, 'spawn_time': 15},
        {'arrival_rate': 2.5, 'service_rate': 1.0, 'num_servers': 2,
         'warm_percent': 0.5, 'spawn_time': 20},
        {'arrival_rate': 3.0, 'service_rate': 1.0, 'num_servers': 2,
         'warm_percent': 0.25, 'spawn_time': 25},
        {'arrival_rate': 1.5, 'service_rate': 0.5, 'num_servers': 2,
         'warm_percent': 0.5, 'spawn_time': 20},
        {'arrival_rate': 4.0, 'service_rate': 2.0, 'num_servers': 2,
         'warm_percent': 0.25, 'spawn_time': 12},
        # Small systems with very long cold starts
        {'arrival_rate': 3.0, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.2, 'spawn_time': 10},
        {'arrival_rate': 3.5, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.2, 'spawn_time': 20},
        {'arrival_rate': 4.0, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.3, 'spawn_time': 8},
        {'arrival_rate': 4.5, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.3, 'spawn_time': 15},
        {'arrival_rate': 2.0, 'service_rate': 0.5, 'num_servers': 3,
         'warm_percent': 0.2, 'spawn_time': 25},
        {'arrival_rate': 5.0, 'service_rate': 2.0, 'num_servers': 3,
         'warm_percent': 0.2, 'spawn_time': 12},
        # Moderate systems with high cold start ratio
        {'arrival_rate': 5.0, 'service_rate': 0.5, 'num_servers': 5,
         'warm_percent': 0.2, 'spawn_time': 12},
        {'arrival_rate': 6.0, 'service_rate': 1.0, 'num_servers': 5,
         'warm_percent': 0.2, 'spawn_time': 15},
        {'arrival_rate': 7.0, 'service_rate': 1.0, 'num_servers': 5,
         'warm_percent': 0.3, 'spawn_time': 20},
        {'arrival_rate': 8.0, 'service_rate': 1.0, 'num_servers': 5,
         'warm_percent': 0.4, 'spawn_time': 15},
        {'arrival_rate': 10.0, 'service_rate': 2.0, 'num_servers': 5,
         'warm_percent': 0.3, 'spawn_time': 10},
        {'arrival_rate': 12.0, 'service_rate': 2.0, 'num_servers': 5,
         'warm_percent': 0.2, 'spawn_time': 25},
        # Near-saturation with asymmetric warm/cold split
        {'arrival_rate': 15.0, 'service_rate': 1.0, 'num_servers': 10,
         'warm_percent': 0.3, 'spawn_time': 8},
        {'arrival_rate': 18.0, 'service_rate': 1.0, 'num_servers': 10,
         'warm_percent': 0.2, 'spawn_time': 15},
        {'arrival_rate': 20.0, 'service_rate': 1.0, 'num_servers': 10,
         'warm_percent': 0.5, 'spawn_time': 6},
        {'arrival_rate': 22.0, 'service_rate': 2.0, 'num_servers': 10,
         'warm_percent': 0.2, 'spawn_time': 20},
        {'arrival_rate': 25.0, 'service_rate': 2.0, 'num_servers': 15,
         'warm_percent': 0.2, 'spawn_time': 10},
        {'arrival_rate': 30.0, 'service_rate': 2.0, 'num_servers': 15,
         'warm_percent': 0.3, 'spawn_time': 15},
        # Heavy overload scenarios
        {'arrival_rate': 10.0, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.3, 'spawn_time': 10},
        {'arrival_rate': 15.0, 'service_rate': 1.0, 'num_servers': 5,
         'warm_percent': 0.3, 'spawn_time': 12},
        {'arrival_rate': 30.0, 'service_rate': 1.0, 'num_servers': 10,
         'warm_percent': 0.2, 'spawn_time': 20},
        {'arrival_rate': 50.0, 'service_rate': 2.0, 'num_servers': 20,
         'warm_percent': 0.2, 'spawn_time': 15},
        # Very slow service + long cold start
        {'arrival_rate': 2.0, 'service_rate': 0.25, 'num_servers': 5,
         'warm_percent': 0.3, 'spawn_time': 20},
        {'arrival_rate': 3.0, 'service_rate': 0.25, 'num_servers': 8,
         'warm_percent': 0.2, 'spawn_time': 25},
        {'arrival_rate': 5.0, 'service_rate': 0.5, 'num_servers': 8,
         'warm_percent': 0.2, 'spawn_time': 20},
        {'arrival_rate': 1.0, 'service_rate': 0.25, 'num_servers': 3,
         'warm_percent': 0.3, 'spawn_time': 25},
    ]

    # Deduplicate extreme cases that already appear in the grid
    grid_keys = set()
    for c in all_cases:
        key = (c['arrival_rate'], c['service_rate'], c['num_servers'],
               c['warm_percent'], c['spawn_time'])
        grid_keys.add(key)

    unique_extreme = []
    for ec in extreme_cases:
        key = (ec['arrival_rate'], ec['service_rate'], ec['num_servers'],
               ec['warm_percent'], ec['spawn_time'])
        if key not in grid_keys:
            unique_extreme.append(ec)
            grid_keys.add(key)

    # Reserve slots for extreme cases, sample the rest from grid
    n_extreme = len(unique_extreme)
    n_grid = min(len(all_cases), num_cases - n_extreme)

    # Shuffle grid cases for diversity (avoid bias toward small num_servers)
    np.random.seed(42)
    indices = np.random.permutation(len(all_cases))
    sampled_grid = [all_cases[i] for i in indices[:n_grid]]

    selected = sampled_grid + unique_extreme

    # Assign case_ids
    for idx, case in enumerate(selected):
        case['case_id'] = idx

    return selected[:num_cases]


def convert_to_erlang_config(test_case):
    """Convert test case to Erlang B model configuration."""

    max_resource = max(cpu_demand, ram_demand)
    total_containers = math.floor(100 / max_resource) * test_case['num_servers']
    queue_warm = int(total_containers * test_case['warm_percent'])
    queue_cold = total_containers - queue_warm
    print(f"Queue warm: {queue_warm}, Queue cold: {queue_cold}")

    config = {
        "lam": test_case['arrival_rate'],
        "mu": test_case['service_rate'],
        "spawn_rate": 1.0 / test_case['spawn_time'],
        "queue_warm": queue_warm,
        "queue_cold": queue_cold,
        "serving_time": "exponential",
        "arrivals": "exponential",
        "ram_warm": ram_warm,
        "cpu_warm": cpu_warm,
        "ram_demand": ram_demand,
        "cpu_demand": cpu_demand,
        "peak_power": 150.0,
        "power_scale": 0.2
    }

    return config


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

    sim_config["request"]["warm_cpu"] = cpu_warm
    sim_config["request"]["warm_ram"] = ram_warm
    sim_config["request"]["cold_start_cpu"] = cpu_warm  # Markov model doesn't distinguish cold_start resource from warm
    sim_config["request"]["cold_start_ram"] = ram_warm
    sim_config["request"]["cpu_demand"] = cpu_demand
    sim_config["request"]["ram_demand"] = ram_demand

    # Ensure distributions match Markov model assumptions (M/M type)
    sim_config["distribution"]["spawn-distribution"] = "exponential"
    sim_config["distribution"]["arrival-distribution"] = "exponential"
    sim_config["distribution"]["service-distribution"] = "exponential"
    return sim_config


# =============================================================================
# Model Runners
# =============================================================================

def run_erlang_b_model(config):
    """Run the cascaded Erlang B baseline model and extract metrics."""
    try:
        model = CascadedErlangBModel(config)
        metrics = model.get_metrics()
        return {
            'blocking_probability': metrics['blocking_probability'],
            'latency': metrics['latency'],
            'cpu_usage': metrics['cpu_usage'],
            'ram_usage': metrics['ram_usage'],
            'power_usage': metrics['power_usage'],
        }
    except Exception as e:
        print(f"  Error in Erlang B model: {e}")
        return None


def run_simulator(sim_config):
    """Run the simulator and extract metrics."""
    try:
        env = simpy.Environment()

        system = System(env, sim_config,
                        distribution=sim_config["distribution"],
                        verbose=sim_config["system"]["verbose"])

        for i in range(sim_config["system"]["num_servers"]):
            server = Server(env, f"Server-{i}", sim_config["server"])
            system.add_server(server)

        pre_warm_done = env.process(system.pre_warm())

        def start_request_generator():
            yield pre_warm_done
            env.process(system.request_generator())

        env.process(start_request_generator())
        env.process(system.resource_monitor_process())

        env.run(until=sim_config["system"]["sim_time"])

        blocking_probability = 0.0
        if system.request_stats['generated'] > 0:
            blocking_probability = (system.request_stats['blocked_no_server_capacity']
                                    / system.request_stats['generated'])

        avg_latency = 0.0
        if system.latency_stats['count'] > 0:
            avg_latency = system.latency_stats['total_latency'] / system.latency_stats['count']

        print(f"  Total latency: {system.latency_stats['total_latency']:.10f}")
        print(f"  Avg latency:   {avg_latency:.10f}")

        mean_cpu_usage = system.get_mean_cpu_usage()
        mean_ram_usage = system.get_mean_ram_usage()
        mean_power_usage = system.get_mean_power_usage()

        return {
            'blocking_probability': blocking_probability,
            'latency': avg_latency,
            'cpu_usage': mean_cpu_usage,
            'ram_usage': mean_ram_usage,
            'power_usage': mean_power_usage,
        }
    except Exception as e:
        print(f"  Error in simulator: {e}")
        return None


# =============================================================================
# Comparison Metrics
# =============================================================================

def calculate_comparison_metrics(erlang_results, sim_results):
    """
    Calculate MAPE, RMSE, NRMSE, and R-squared.
    Simulator is treated as the reference (ground truth).
    """
    metrics = ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']
    comparison_results = {}

    for metric in metrics:
        erlang_vals = np.array([r[metric] for r in erlang_results if r is not None])
        sim_vals = np.array([r[metric] for r in sim_results if r is not None])

        min_len = min(len(erlang_vals), len(sim_vals))
        erlang_vals = erlang_vals[:min_len]
        sim_vals = sim_vals[:min_len]

        if len(sim_vals) == 0:
            comparison_results[metric] = {
                'MAPE': float('inf'), 'RMSE': float('inf'),
                'NRMSE': float('inf'), 'R_squared': 0.0,
                'mean_erlang': 0.0, 'mean_sim': 0.0,
            }
            continue

        # MAPE (Erlang B vs Simulator reference)
        if metric == 'blocking_probability':
            threshold = 0.01
            if np.mean(sim_vals) < threshold:
                mape = np.mean(np.abs(erlang_vals - sim_vals)) * 100
            else:
                mape = np.mean(np.abs((erlang_vals - sim_vals) / np.maximum(sim_vals, 1e-10))) * 100
        else:
            mape = np.mean(np.abs((erlang_vals - sim_vals) / np.maximum(sim_vals, 1e-10))) * 100

        # RMSE
        rmse = np.sqrt(np.mean((erlang_vals - sim_vals) ** 2))

        # Normalized RMSE
        nrmse = (rmse / np.mean(sim_vals)) * 100 if np.mean(sim_vals) > 0 else float('inf')

        # R-squared (using Simulator as reference)
        ss_res = np.sum((erlang_vals - sim_vals) ** 2)
        ss_tot = np.sum((sim_vals - np.mean(sim_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Signed mean error (bias): positive means Erlang B overestimates
        mean_error = np.mean(erlang_vals - sim_vals)

        comparison_results[metric] = {
            'MAPE': mape,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'R_squared': r_squared,
            'mean_error': mean_error,
            'mean_erlang': np.mean(erlang_vals),
            'mean_sim': np.mean(sim_vals),
        }

    return comparison_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  Cascaded Erlang B Baseline  vs  Simulator")
    print("=" * 70)

    test_cases = generate_test_cases(500)
    print(f"Generated {len(test_cases)} test cases\n")

    erlang_results = []
    sim_results = []
    detailed_results = []

    for i, tc in enumerate(test_cases):
        print(f"\nTest case {i+1}/{len(test_cases)}  |  "
              f"λ={tc['arrival_rate']:.1f}, μ={tc['service_rate']:.2f}, "
              f"servers={tc['num_servers']}, warm%={tc['warm_percent']:.1f}, "
              f"spawn_t={tc['spawn_time']}")

        erlang_config = convert_to_erlang_config(tc)
        sim_config = convert_to_simulator_config(tc)

        # Run Erlang B model
        print("  Running Erlang B model...")
        erlang_result = run_erlang_b_model(erlang_config)

        # Run simulator
        print("  Running simulator...")
        sim_result = run_simulator(sim_config)

        if erlang_result is not None and sim_result is not None:
            erlang_results.append(erlang_result)
            sim_results.append(sim_result)

            detailed_results.append({
                'case_id': tc['case_id'],
                'arrival_rate': tc['arrival_rate'],
                'service_rate': tc['service_rate'],
                'num_servers': tc['num_servers'],
                'warm_percent': tc['warm_percent'],
                'spawn_time': tc['spawn_time'],
                'n_warm': erlang_config['queue_warm'],
                'n_cold': erlang_config['queue_cold'],
                # Erlang B
                'erlang_blocking': erlang_result['blocking_probability'],
                'erlang_latency': erlang_result['latency'],
                'erlang_cpu': erlang_result['cpu_usage'],
                'erlang_ram': erlang_result['ram_usage'],
                'erlang_power': erlang_result['power_usage'],
                # Simulator
                'sim_blocking': sim_result['blocking_probability'],
                'sim_latency': sim_result['latency'],
                'sim_cpu': sim_result['cpu_usage'],
                'sim_ram': sim_result['ram_usage'],
                'sim_power': sim_result['power_usage'],
            })

            print(f"  ErlangB:  block={erlang_result['blocking_probability']:.6f}, "
                  f"latency={erlang_result['latency']:.4f}, "
                  f"cpu={erlang_result['cpu_usage']:.2f}, ram={erlang_result['ram_usage']:.2f}, "
                  f"power={erlang_result['power_usage']:.2f}")
            print(f"  Sim:      block={sim_result['blocking_probability']:.6f}, "
                  f"latency={sim_result['latency']:.4f}, "
                  f"cpu={sim_result['cpu_usage']:.2f}, ram={sim_result['ram_usage']:.2f}, "
                  f"power={sim_result['power_usage']:.2f}")
        else:
            print("  ERROR: One or both models failed!")

    # -------------------------------------------------------------------------
    # Comparison Metrics
    # -------------------------------------------------------------------------
    print(f"\n\nCalculating comparison metrics for {len(erlang_results)} successful test cases...")
    comparison = calculate_comparison_metrics(erlang_results, sim_results)

    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS  (Erlang B deviation from Simulator reference)")
    print("=" * 70)

    for metric in ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        m = comparison[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"    MAPE:       {m['MAPE']:.2f}%")
        print(f"    RMSE:       {m['RMSE']:.6f}")
        print(f"    NRMSE:      {m['NRMSE']:.2f}%")
        print(f"    R-squared:  {m['R_squared']:.4f}")
        print(f"    Mean Error: {m['mean_error']:+.6f}  "
              f"(ErlangB avg={m['mean_erlang']:.6f}, Sim avg={m['mean_sim']:.6f})")

    # -------------------------------------------------------------------------
    # Bias & Error Distribution Analysis (Erlang B)
    # -------------------------------------------------------------------------
    if len(detailed_results) > 0:
        df_detail = pd.DataFrame(detailed_results)

        print(f"\n{'=' * 70}")
        print(f"  BIAS & ERROR DISTRIBUTION ANALYSIS (Erlang B vs Simulator)")
        print(f"{'=' * 70}")

        # Per-metric signed error analysis
        for metric_short, metric_label in [
            ('blocking', 'Blocking Prob.'), ('latency', 'Latency'),
            ('cpu', 'CPU Usage'), ('ram', 'RAM Usage'), ('power', 'Power Usage'),
        ]:
            model_col = f'erlang_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df_detail.columns or sim_col not in df_detail.columns:
                continue

            errors = df_detail[model_col] - df_detail[sim_col]
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

        # --- Conditional analysis by n_warm ---
        print(f"\n  {'─' * 60}")
        print(f"  Conditional MAPE by n_warm (warm pool size)")
        print(f"  {'─' * 60}")

        bins = [0, 2, 4, 8, 16, 100]
        labels = ['1-2', '3-4', '5-8', '9-16', '17+']
        df_detail['n_warm_bin'] = pd.cut(df_detail['n_warm'], bins=bins, labels=labels, right=True)

        for metric_short, metric_label in [
            ('blocking', 'Blocking'), ('latency', 'Latency'),
            ('cpu', 'CPU'), ('ram', 'RAM'), ('power', 'Power'),
        ]:
            model_col = f'erlang_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df_detail.columns or sim_col not in df_detail.columns:
                continue

            print(f"\n    {metric_label}:")
            for bin_label in labels:
                subset = df_detail[df_detail['n_warm_bin'] == bin_label]
                if len(subset) == 0:
                    continue
                errs = subset[model_col] - subset[sim_col]
                sim_vals = subset[sim_col]
                if sim_vals.mean() > 0.01:
                    mape = (errs.abs() / sim_vals.replace(0, np.nan)).mean() * 100
                else:
                    mape = errs.abs().mean() * 100
                bias = errs.mean()
                print(f"      n_warm={bin_label:>5s}  (n={len(subset):3d}):  "
                      f"MAPE={mape:6.2f}%,  bias={bias:+.6f}")

        # --- Conditional analysis by spawn_time ---
        print(f"\n  {'─' * 60}")
        print(f"  Conditional MAPE by spawn_time")
        print(f"  {'─' * 60}")

        for metric_short, metric_label in [
            ('blocking', 'Blocking'), ('latency', 'Latency'),
        ]:
            model_col = f'erlang_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df_detail.columns or sim_col not in df_detail.columns:
                continue

            print(f"\n    {metric_label}:")
            for st in sorted(df_detail['spawn_time'].unique()):
                subset = df_detail[df_detail['spawn_time'] == st]
                if len(subset) == 0:
                    continue
                errs = subset[model_col] - subset[sim_col]
                sim_vals = subset[sim_col]
                if sim_vals.mean() > 0.01:
                    mape = (errs.abs() / sim_vals.replace(0, np.nan)).mean() * 100
                else:
                    mape = errs.abs().mean() * 100
                bias = errs.mean()
                print(f"      spawn_t={st:>3}  (n={len(subset):3d}):  "
                      f"MAPE={mape:6.2f}%,  bias={bias:+.6f}")

        # --- Conditional analysis by load regime ---
        print(f"\n  {'─' * 60}")
        print(f"  Conditional MAPE by load regime (λ / effective capacity)")
        print(f"  {'─' * 60}")

        def _compute_rho(row):
            cps = math.floor(100 / max(cpu_demand, ram_demand))
            nw = int(cps * row['num_servers'] * row['warm_percent'])
            nc = cps * row['num_servers'] - nw
            warm_cap = nw * row['service_rate']
            cold_eff = 1.0 / (row['spawn_time'] + 1.0 / row['service_rate'])
            total_cap = warm_cap + nc * cold_eff
            return row['arrival_rate'] / total_cap if total_cap > 0 else float('inf')

        df_detail['rho'] = df_detail.apply(_compute_rho, axis=1)

        rho_bins = [0, 0.7, 1.0, 1.3, 100]
        rho_labels = ['light(<0.7)', 'moderate(0.7-1.0)', 'heavy(1.0-1.3)', 'overload(>1.3)']
        df_detail['load_regime'] = pd.cut(df_detail['rho'], bins=rho_bins, labels=rho_labels, right=True)

        for metric_short, metric_label in [
            ('blocking', 'Blocking'), ('latency', 'Latency'),
        ]:
            model_col = f'erlang_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df_detail.columns or sim_col not in df_detail.columns:
                continue

            print(f"\n    {metric_label}:")
            for regime in rho_labels:
                subset = df_detail[df_detail['load_regime'] == regime]
                if len(subset) == 0:
                    continue
                errs = subset[model_col] - subset[sim_col]
                sim_vals = subset[sim_col]
                if sim_vals.mean() > 0.01:
                    mape = (errs.abs() / sim_vals.replace(0, np.nan)).mean() * 100
                else:
                    mape = errs.abs().mean() * 100
                bias = errs.mean()
                print(f"      {regime:>20s}  (n={len(subset):3d}):  "
                      f"MAPE={mape:6.2f}%,  bias={bias:+.6f}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total test cases:          {len(test_cases)}")
    # print(f"  Successful comparisons:    {len(erlang_results)}")
    # print(f"  Success rate:              {len(erlang_results)/len(test_cases)*100:.1f}%")
    # print(f"\n  The cascaded Erlang B baseline assumes independent pools and")
    # print(f"  Poisson overflow. Comparing against the simulator reveals where")
    # print(f"  this approximation breaks down in practice.")


if __name__ == "__main__":
    main()
