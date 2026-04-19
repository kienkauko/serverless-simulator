#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script between Markov Model and Simulator
Runs both systems with the same input parameters and compares their results
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
from scipy import stats
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

# Fixed resource values (matching simulator defaults)
cpu_warm = 1
ram_warm = 30
cpu_demand = 50
ram_demand = 40

def generate_test_cases(num_cases=100):
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

def convert_to_markov_config(test_case):
    """Convert test case to Markov model configuration"""
    
    # Calculate queue sizes based on simulator logic
    max_resource = max(cpu_demand, ram_demand)
    total_containers = math.floor(100 / max_resource) * test_case['num_servers']
    queue_warm = int(total_containers * test_case['warm_percent'])
    queue_cold = total_containers - queue_warm
    print(f"Queue warm: {queue_warm}, Queue cold: {queue_cold}")
    markov_config = {
        "lam": test_case['arrival_rate'],
        "mu": test_case['service_rate'],
        "spawn_rate": 1.0 / test_case['spawn_time'],
        "queue_warm": queue_warm,
        "queue_cold": queue_cold,
        "serving_time": "exponential",
        "arrivals": "exponential",
        "ram_warm": ram_warm,
        "cpu_warm": cpu_warm,
        "ram_cold": ram_warm,
        "cpu_cold": cpu_warm,
        "ram_demand": ram_demand,
        "cpu_demand": cpu_demand,
        "peak_power": 150.0,
        "power_scale": 0.2
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
        # Reset global statistics
        # global request_stats, latency_stats
        # request_stats = {
        #     'generated': 0, 'processed': 0, 'blocked_no_server_capacity': 0, 
        #     'blocked_spawn_failed': 0, 'blocked_no_path': 0, 'container_spawns_initiated': 0,
        #     'container_spawns_failed': 0, 'container_spawns_succeeded': 0, 'containers_reused': 0,
        #     'containers_removed_idle': 0, 'reuse_oom_failures': 0
        # }
        
        # latency_stats = {
        #     'total_latency': 0.0, 'spawning_time': 0.0, 'processing_time': 0.0,
        #     'waiting_time': 0.0, 'container_wait_time': 0.0, 'assignment_time': 0.0, 'count': 0
        # }
        
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
        print(f"Total latency: {system.latency_stats['total_latency']:.10f}")
        print(f"Avg latency: {avg_latency:.10f}")
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

    print("Starting Markov Model vs Simulator Comparison")
    print(f"  Simulation replications per scenario: {NUM_SIM_REPLICATIONS}")
    print(f"  Confidence level: {CI_LEVEL*100:.0f}%")
    print("=" * 60)

    # Generate test cases
    test_cases = generate_test_cases(100)
    print(f"Generated {len(test_cases)} test cases")

    # Store results
    markov_results = []
    sim_results = []  # stores the mean across replications
    detailed_results = []

    metrics_list = ['blocking_probability', 'latency', 'variance_latency',
                    'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']

    # Run comparisons
    for i, test_case in enumerate(test_cases):
        print(f"\nRunning test case {i+1}/{len(test_cases)}")
        print(f"Parameters: λ={test_case['arrival_rate']:.1f}, μ={test_case['service_rate']:.1f}, "
              f"servers={test_case['num_servers']}, warm%={test_case['warm_percent']:.1f}, "
              f"spawn_time={test_case['spawn_time']}")

        # Convert to respective configurations
        markov_config = convert_to_markov_config(test_case)
        sim_config = convert_to_simulator_config(test_case)

        # Run Markov model (deterministic — only once)
        print("  Running Markov model...")
        markov_result = run_markov_model(markov_config)

        # Run simulator NUM_SIM_REPLICATIONS times
        print(f"  Running simulator ({NUM_SIM_REPLICATIONS} replications)...")
        sim_replication_results = []
        for _ in range(NUM_SIM_REPLICATIONS):
            res = run_simulator(sim_config)
            if res is not None:
                sim_replication_results.append(res)

        if markov_result is None or len(sim_replication_results) < 2:
            print("  ERROR: Markov failed or insufficient sim replications!")
            continue

        # Compute mean and 95% CI for each metric from sim replications
        n_reps = len(sim_replication_results)
        t_crit = stats.t.ppf((1 + CI_LEVEL) / 2, df=n_reps - 1)

        sim_mean = {}
        ci_lower = {}
        ci_upper = {}
        ci_coverage = {}  # whether Markov falls inside CI

        for metric in metrics_list:
            vals = np.array([r[metric] for r in sim_replication_results])
            mean_val = np.mean(vals)
            std_val = np.std(vals, ddof=1)
            margin = t_crit * std_val / np.sqrt(n_reps)

            sim_mean[metric] = mean_val
            ci_lower[metric] = mean_val - margin
            ci_upper[metric] = mean_val + margin
            ci_coverage[metric] = (ci_lower[metric] <= markov_result[metric] <= ci_upper[metric])

        markov_results.append(markov_result)
        sim_results.append(sim_mean)

        # Store detailed results
        row = {
            'case_id': test_case['case_id'],
            'arrival_rate': test_case['arrival_rate'],
            'service_rate': test_case['service_rate'],
            'num_servers': test_case['num_servers'],
            'warm_percent': test_case['warm_percent'],
            'spawn_time': test_case['spawn_time'],
            'n_warm': markov_config['queue_warm'],
            'n_cold': markov_config['queue_cold'],
            'n_reps': n_reps,
        }
        for metric in metrics_list:
            short = metric
            row[f'markov_{short}'] = markov_result[metric]
            row[f'sim_mean_{short}'] = sim_mean[metric]
            row[f'ci_lower_{short}'] = ci_lower[metric]
            row[f'ci_upper_{short}'] = ci_upper[metric]
            row[f'ci_cover_{short}'] = ci_coverage[metric]
        detailed_results.append(row)

        # Print per-case summary
        print(f"  Markov: block={markov_result['blocking_probability']:.4f}, "
              f"latency={markov_result['latency']:.4f}, "
              f"var={markov_result['variance_latency']:.4f}, "
              f"p99={markov_result['p99_latency']:.4f}, "
              f"cpu={markov_result['cpu_usage']:.2f}, ram={markov_result['ram_usage']:.2f}, "
              f"power={markov_result['power_usage']:.2f}")
        print(f"  Sim μ:  block={sim_mean['blocking_probability']:.4f}, "
              f"latency={sim_mean['latency']:.4f}, "
              f"var={sim_mean['variance_latency']:.4f}, "
              f"p99={sim_mean['p99_latency']:.4f}, "
              f"cpu={sim_mean['cpu_usage']:.2f}, ram={sim_mean['ram_usage']:.2f}, "
              f"power={sim_mean['power_usage']:.2f}")
        cover_str = ", ".join(
            f"{m.split('_')[0][:3]}={'Y' if ci_coverage[m] else 'N'}"
            for m in metrics_list
        )
        print(f"  CI coverage: {cover_str}")
    
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
    # 95% CI Coverage Analysis
    # =========================================================================
    if len(detailed_results) > 0:
        df = pd.DataFrame(detailed_results)
        n_scenarios = len(df)

        print(f"\n{'=' * 70}")
        print(f"  95% CONFIDENCE INTERVAL COVERAGE ANALYSIS")
        print(f"  (Markov prediction inside simulator {CI_LEVEL*100:.0f}% CI?)")
        print(f"  Scenarios: {n_scenarios},  Replications per scenario: {NUM_SIM_REPLICATIONS}")
        print(f"{'=' * 70}")

        for metric in metrics_list:
            cover_col = f'ci_cover_{metric}'
            if cover_col not in df.columns:
                continue
            n_covered = df[cover_col].sum()
            pct = 100 * n_covered / n_scenarios
            print(f"\n  {metric.upper().replace('_', ' ')}:")
            print(f"    Covered: {n_covered}/{n_scenarios} ({pct:.1f}%)")

        # --- Conditional CI coverage by n_warm ---
        print(f"\n  {'─' * 60}")
        print(f"  CI Coverage by n_warm (warm pool size)")
        print(f"  {'─' * 60}")

        bins = [0, 2, 4, 8, 16, 100]
        bin_labels = ['1-2', '3-4', '5-8', '9-16', '17+']
        df['n_warm_bin'] = pd.cut(df['n_warm'], bins=bins, labels=bin_labels, right=True)

        for metric in metrics_list:
            cover_col = f'ci_cover_{metric}'
            if cover_col not in df.columns:
                continue
            print(f"\n    {metric}:")
            for bl in bin_labels:
                subset = df[df['n_warm_bin'] == bl]
                if len(subset) == 0:
                    continue
                nc = subset[cover_col].sum()
                print(f"      n_warm={bl:>5s}  (n={len(subset):3d}):  "
                      f"covered={nc}/{len(subset)} ({100*nc/len(subset):.1f}%)")

        # --- Conditional CI coverage by spawn_time ---
        print(f"\n  {'─' * 60}")
        print(f"  CI Coverage by spawn_time")
        print(f"  {'─' * 60}")

        for metric in metrics_list:
            cover_col = f'ci_cover_{metric}'
            if cover_col not in df.columns:
                continue
            print(f"\n    {metric}:")
            for st in sorted(df['spawn_time'].unique()):
                subset = df[df['spawn_time'] == st]
                if len(subset) == 0:
                    continue
                nc = subset[cover_col].sum()
                print(f"      spawn_t={st:>3}  (n={len(subset):3d}):  "
                      f"covered={nc}/{len(subset)} ({100*nc/len(subset):.1f}%)")

        # --- Conditional CI coverage by load regime ---
        print(f"\n  {'─' * 60}")
        print(f"  CI Coverage by load regime (λ / effective capacity)")
        print(f"  {'─' * 60}")

        def _compute_rho(row):
            cps = math.floor(100 / max(cpu_demand, ram_demand))
            nw = int(cps * row['num_servers'] * row['warm_percent'])
            nc = cps * row['num_servers'] - nw
            warm_cap = nw * row['service_rate']
            cold_eff = 1.0 / (row['spawn_time'] + 1.0 / row['service_rate'])
            total_cap = warm_cap + nc * cold_eff
            return row['arrival_rate'] / total_cap if total_cap > 0 else float('inf')

        df['rho'] = df.apply(_compute_rho, axis=1)

        rho_bins = [0, 0.7, 1.0, 1.3, 100]
        rho_labels = ['light(<0.7)', 'moderate(0.7-1.0)', 'heavy(1.0-1.3)', 'overload(>1.3)']
        df['load_regime'] = pd.cut(df['rho'], bins=rho_bins, labels=rho_labels, right=True)

        for metric in metrics_list:
            cover_col = f'ci_cover_{metric}'
            if cover_col not in df.columns:
                continue
            print(f"\n    {metric}:")
            for regime in rho_labels:
                subset = df[df['load_regime'] == regime]
                if len(subset) == 0:
                    continue
                nc = subset[cover_col].sum()
                print(f"      {regime:>20s}  (n={len(subset):3d}):  "
                      f"covered={nc}/{len(subset)} ({100*nc/len(subset):.1f}%)")

    # -------------------------------------------------------------------------
    # Bias & Error Distribution Analysis (CTMC vs Sim mean)
    # -------------------------------------------------------------------------
    if len(detailed_results) > 0:
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
    filename = f"comparison_results/markov_vs_sim_{timestamp}.csv"

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

if __name__ == "__main__":
    main()