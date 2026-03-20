#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script between Markov Model and Simulator (trace-based service)

Differences from CTMC_vs_simulator.py:
- Service rates are NOT swept across cases.
- Simulator uses service-distribution = "traces".
- Markov mu is fixed from trace mean service time (48.95 ms).
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

# Import Markov model
from Markov.model_2D import MarkovModel

# Import simulator components
from variables import config as base_config
from Server import Server
from System import System

# Fixed resource values (matching simulator defaults)
cpu_warm = 1
ram_warm = 30
cpu_demand = 50
ram_demand = 40

# Trace-derived service settings
TRACE_MEAN_SERVICE_MS = 48.95
TRACE_MEAN_SERVICE_S = TRACE_MEAN_SERVICE_MS / 1000.0
TRACE_MU = 1.0 / TRACE_MEAN_SERVICE_S  # req/s, used by CTMC


def generate_test_cases(num_cases=200):
    """Generate test cases with fixed service rate derived from traces.

    Excludes service_rate sweep; uses a single fixed value TRACE_MU.
    """
    all_cases = []

    max_resource = max(cpu_demand, ram_demand)
    containers_per_server = math.floor(100 / max_resource)  # = 2

    # Expanded parameter grids (without service_rates)
    num_servers_list = [1, 2, 3, 4, 5, 8, 10, 15, 20]
    warm_percents = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    spawn_times = [1, 2, 4, 6, 8, 12, 16, 20, 25]
    load_fractions = [0.5, 0.7, 0.85, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]

    service_rate = TRACE_MU

    for num_servers in num_servers_list:
        total_cap = containers_per_server * num_servers
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

    # Extreme coupling cases (service_rate fixed to TRACE_MU)
    extreme_cases = [
        {'arrival_rate': 0.8, 'service_rate': service_rate, 'num_servers': 1, 'warm_percent': 0.5, 'spawn_time': 10},
        {'arrival_rate': 1.2, 'service_rate': service_rate, 'num_servers': 1, 'warm_percent': 0.5, 'spawn_time': 15},
        {'arrival_rate': 1.5, 'service_rate': service_rate, 'num_servers': 1, 'warm_percent': 0.5, 'spawn_time': 20},
        {'arrival_rate': 2.0, 'service_rate': service_rate, 'num_servers': 2, 'warm_percent': 0.25, 'spawn_time': 15},
        {'arrival_rate': 2.5, 'service_rate': service_rate, 'num_servers': 2, 'warm_percent': 0.5, 'spawn_time': 20},
        {'arrival_rate': 3.0, 'service_rate': service_rate, 'num_servers': 2, 'warm_percent': 0.25, 'spawn_time': 25},
        {'arrival_rate': 3.0, 'service_rate': service_rate, 'num_servers': 3, 'warm_percent': 0.2, 'spawn_time': 10},
        {'arrival_rate': 3.5, 'service_rate': service_rate, 'num_servers': 3, 'warm_percent': 0.2, 'spawn_time': 20},
        {'arrival_rate': 4.5, 'service_rate': service_rate, 'num_servers': 3, 'warm_percent': 0.3, 'spawn_time': 15},
        {'arrival_rate': 6.0, 'service_rate': service_rate, 'num_servers': 5, 'warm_percent': 0.2, 'spawn_time': 15},
        {'arrival_rate': 8.0, 'service_rate': service_rate, 'num_servers': 5, 'warm_percent': 0.4, 'spawn_time': 15},
        {'arrival_rate': 15.0, 'service_rate': service_rate, 'num_servers': 10, 'warm_percent': 0.3, 'spawn_time': 8},
        {'arrival_rate': 18.0, 'service_rate': service_rate, 'num_servers': 10, 'warm_percent': 0.2, 'spawn_time': 15},
        {'arrival_rate': 25.0, 'service_rate': service_rate, 'num_servers': 15, 'warm_percent': 0.2, 'spawn_time': 10},
        {'arrival_rate': 30.0, 'service_rate': service_rate, 'num_servers': 15, 'warm_percent': 0.3, 'spawn_time': 15},
        {'arrival_rate': 50.0, 'service_rate': service_rate, 'num_servers': 20, 'warm_percent': 0.2, 'spawn_time': 15},
    ]

    # Deduplicate extreme cases that already appear in the grid
    grid_keys = set()
    for c in all_cases:
        key = (c['arrival_rate'], c['service_rate'], c['num_servers'], c['warm_percent'], c['spawn_time'])
        grid_keys.add(key)

    unique_extreme = []
    for ec in extreme_cases:
        key = (ec['arrival_rate'], ec['service_rate'], ec['num_servers'], ec['warm_percent'], ec['spawn_time'])
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


def convert_to_markov_config(test_case):
    """Convert test case to Markov model configuration."""
    max_resource = max(cpu_demand, ram_demand)
    total_containers = math.floor(100 / max_resource) * test_case['num_servers']
    queue_warm = int(total_containers * test_case['warm_percent'])
    queue_cold = total_containers - queue_warm
    print(f"Queue warm: {queue_warm}, Queue cold: {queue_cold}")

    markov_config = {
        "lam": test_case['arrival_rate'],
        "mu": TRACE_MU,
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
        "power_scale": 0.2,
    }

    return markov_config


def convert_to_simulator_config(test_case):
    """Convert test case to simulator configuration."""
    sim_config = copy.deepcopy(base_config)

    sim_config["system"]["num_servers"] = test_case['num_servers']
    sim_config["system"]["warm_percent"] = test_case['warm_percent']
    sim_config["request"]["arrival_rate_mean"] = test_case['arrival_rate']
    sim_config["request"]["arrival_rate_std"] = 0
    sim_config["request"]["service_rate"] = TRACE_MU
    sim_config["container"]["spawn_time_mean"] = test_case['spawn_time']
    sim_config["container"]["spawn_time_std"] = 0

    sim_config["request"]["warm_cpu"] = cpu_warm
    sim_config["request"]["warm_ram"] = ram_warm
    sim_config["request"]["cold_start_cpu"] = cpu_warm
    sim_config["request"]["cold_start_ram"] = ram_warm
    sim_config["request"]["cpu_demand"] = cpu_demand
    sim_config["request"]["ram_demand"] = ram_demand

    sim_config["distribution"]["spawn-distribution"] = "exponential"
    sim_config["distribution"]["arrival-distribution"] = "exponential"
    sim_config["distribution"]["service-distribution"] = "traces"

    return sim_config


def run_markov_model(markov_config):
    """Run the Markov model and extract metrics."""
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
    """Run the simulator and extract metrics."""
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
        env.process(system.resource_monitor_process())

        env.run(until=sim_config["system"]["sim_time"])

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
            'power_usage': mean_power_usage,
        }

    except Exception as e:
        print(f"Error in simulator: {e}")
        return None


def calculate_comparison_metrics(markov_results, sim_results):
    """Calculate MAPE, RMSE, and R-squared for comparison."""
    metrics = ['blocking_probability', 'latency', 'variance_latency', 'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']
    comparison_results = {}

    for metric in metrics:
        markov_vals = np.array([r[metric] for r in markov_results if r is not None])
        sim_vals = np.array([r[metric] for r in sim_results if r is not None])

        min_len = min(len(markov_vals), len(sim_vals))
        markov_vals = markov_vals[:min_len]
        sim_vals = sim_vals[:min_len]

        if len(markov_vals) == 0:
            comparison_results[metric] = {'MAPE': float('inf'), 'RMSE': float('inf'), 'R_squared': 0.0}
            continue

        if metric == 'blocking_probability':
            threshold = 0.01
            if np.mean(sim_vals) < threshold:
                mape = np.mean(np.abs(markov_vals - sim_vals)) * 100
            else:
                mape = np.mean(np.abs((markov_vals - sim_vals) / np.maximum(sim_vals, 1e-10))) * 100
        else:
            mape = np.mean(np.abs((markov_vals - sim_vals) / np.maximum(sim_vals, 1e-10))) * 100

        rmse = np.sqrt(np.mean((markov_vals - sim_vals) ** 2))
        nrmse = (rmse / np.mean(sim_vals)) * 100 if np.mean(sim_vals) > 0 else float('inf')

        ss_res = np.sum((markov_vals - sim_vals) ** 2)
        ss_tot = np.sum((sim_vals - np.mean(sim_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

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
    """Main comparison function."""
    print("Starting Markov Model vs Simulator Comparison (trace-based service)")
    print("=" * 60)
    print(f"Fixed trace mean service: {TRACE_MEAN_SERVICE_MS} ms => mu={TRACE_MU:.6f} req/s")

    test_cases = generate_test_cases(200)
    print(f"Generated {len(test_cases)} test cases")

    markov_results = []
    sim_results = []
    detailed_results = []

    for i, test_case in enumerate(test_cases):
        print(f"\nRunning test case {i+1}/{len(test_cases)}")
        print(f"Parameters: λ={test_case['arrival_rate']:.2f}, μ={TRACE_MU:.2f}, "
              f"servers={test_case['num_servers']}, warm%={test_case['warm_percent']:.1f}, "
              f"spawn_time={test_case['spawn_time']}")

        markov_config = convert_to_markov_config(test_case)
        sim_config = convert_to_simulator_config(test_case)

        print("  Running Markov model...")
        markov_result = run_markov_model(markov_config)

        print("  Running simulator...")
        sim_result = run_simulator(sim_config)

        if markov_result is not None and sim_result is not None:
            markov_results.append(markov_result)
            sim_results.append(sim_result)

            detailed_results.append({
                'case_id': test_case['case_id'],
                'arrival_rate': test_case['arrival_rate'],
                'service_rate': TRACE_MU,
                'num_servers': test_case['num_servers'],
                'warm_percent': test_case['warm_percent'],
                'spawn_time': test_case['spawn_time'],
                'n_warm': markov_config['queue_warm'],
                'n_cold': markov_config['queue_cold'],
                'markov_blocking': markov_result['blocking_probability'],
                'sim_blocking': sim_result['blocking_probability'],
                'markov_latency': markov_result['latency'],
                'sim_latency': sim_result['latency'],
                'markov_variance_latency': markov_result['variance_latency'],
                'sim_variance_latency': sim_result['variance_latency'],
                'markov_p99_latency': markov_result['p99_latency'],
                'sim_p99_latency': sim_result['p99_latency'],
                'markov_cpu': markov_result['cpu_usage'],
                'sim_cpu': sim_result['cpu_usage'],
                'markov_ram': markov_result['ram_usage'],
                'sim_ram': sim_result['ram_usage'],
                'markov_power': markov_result['power_usage'],
                'sim_power': sim_result['power_usage'],
            })

            print(f"  Markov: block={markov_result['blocking_probability']:.4f}, "
                  f"latency={markov_result['latency']:.4f}, "
                  f"var={markov_result['variance_latency']:.4f}, "
                  f"p99={markov_result['p99_latency']:.4f}, "
                  f"cpu={markov_result['cpu_usage']:.2f}, ram={markov_result['ram_usage']:.2f}, "
                  f"power={markov_result['power_usage']:.2f}")
            print(f"  Sim:    block={sim_result['blocking_probability']:.4f}, "
                  f"latency={sim_result['latency']:.4f}, "
                  f"var={sim_result['variance_latency']:.4f}, "
                  f"p99={sim_result['p99_latency']:.4f}, "
                  f"cpu={sim_result['cpu_usage']:.2f}, ram={sim_result['ram_usage']:.2f}, "
                  f"power={sim_result['power_usage']:.2f}")
        else:
            print("  ERROR: One or both models failed!")

    print(f"\n\nCalculating comparison metrics for {len(markov_results)} successful test cases...")
    comparison_metrics = calculate_comparison_metrics(markov_results, sim_results)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    for metric in ['blocking_probability', 'latency', 'variance_latency', 'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        m = comparison_metrics[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"    MAPE:       {m['MAPE']:.2f}%")
        print(f"    RMSE:       {m['RMSE']:.6f}")
        print(f"    NRMSE:      {m['NRMSE']:.2f}%")
        print(f"    R-squared:  {m['R_squared']:.4f}")
        print(f"    Mean Error: {m['mean_error']:+.6f}  "
              f"(CTMC avg={m['mean_markov']:.6f}, Sim avg={m['mean_sim']:.6f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results/markov_vs_sim_traces_{timestamp}.csv"
    os.makedirs("comparison_results", exist_ok=True)

    combined_results = []
    for test_case, markov_result in zip(test_cases[:len(markov_results)], markov_results):
        combined_results.append({
            'case_id': test_case['case_id'],
            'model_type': 'Markov',
            'arrival_rate': test_case['arrival_rate'],
            'service_rate': TRACE_MU,
            'num_servers': test_case['num_servers'],
            'warm_percent': test_case['warm_percent'],
            'spawn_time': test_case['spawn_time'],
            'blocking_probability': markov_result['blocking_probability'],
            'latency': markov_result['latency'],
            'variance_latency': markov_result['variance_latency'],
            'p99_latency': markov_result['p99_latency'],
            'cpu_usage': markov_result['cpu_usage'],
            'ram_usage': markov_result['ram_usage'],
        })

    for test_case, sim_result in zip(test_cases[:len(sim_results)], sim_results):
        combined_results.append({
            'case_id': test_case['case_id'],
            'model_type': 'Simulator',
            'arrival_rate': test_case['arrival_rate'],
            'service_rate': TRACE_MU,
            'num_servers': test_case['num_servers'],
            'warm_percent': test_case['warm_percent'],
            'spawn_time': test_case['spawn_time'],
            'blocking_probability': sim_result['blocking_probability'],
            'latency': sim_result['latency'],
            'variance_latency': sim_result['variance_latency'],
            'p99_latency': sim_result['p99_latency'],
            'cpu_usage': sim_result['cpu_usage'],
            'ram_usage': sim_result['ram_usage'],
        })

    pd.DataFrame(combined_results).to_csv(filename, index=False)
    print(f"\nCombined results saved to: {filename}")

    print(f"\nSUMMARY:")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful comparisons: {len(markov_results)}")
    print(f"Success rate: {len(markov_results)/len(test_cases)*100:.1f}%")


if __name__ == "__main__":
    main()
