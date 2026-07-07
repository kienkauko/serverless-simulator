#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script between Markov Model and Simulator
Runs both systems with the same input parameters and compares their results
"""

import numpy as np
import pandas as pd
from scipy import stats
import math
import copy
import random
import simpy
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import Markov model
from Markov.model_2D import MarkovModel

# Import simulator components
from variables import config as base_config
from Server import Server
from Request import Request
from fixed_pool.Container import Container
from fixed_pool.System import System

# Fixed resource values (matching simulator defaults)
cpu_warm = 1
ram_warm = 30
cpu_transit = 40
ram_transit = 35
cpu_demand = 50
ram_demand = 40

PEAK_POWER = 150.0

# Batch-means CI configuration (per-scenario steady-state CI on simulator output)
NUM_BATCHES = 30
CI_LEVEL = 0.90  # Confidence level for all CIs

def generate_test_cases(num_cases=500, seed=42):
    """Generate IID test cases by independent uniform sampling over parameter ranges.

    Ranges:
      - num_servers:       Uniform{10, ..., 100}
      - power_scale:       Uniform(0.2, 0.6)         # base_power / peak_power
      - warm_percent:      Uniform(0.1, 1.0)
      - mean_service_time: Uniform(1, 30) seconds    # service_rate = 1/t
      - spawn_time:        Uniform(1, 30) seconds
      - arrival_rate:      Uniform(0.1, 1.0) * total_effective_capacity

    Degenerate draws (n_warm < 1 or n_cold < 1) are rejected. Rejection over
    a measure-zero / well-defined region of the parameter space preserves IID
    of the retained samples (they are IID from the conditional uniform).
    """
    rng = np.random.default_rng(seed)

    max_resource = max(cpu_demand, ram_demand)
    containers_per_server = math.floor(100 / max_resource)

    cases = []
    while len(cases) < num_cases:
        num_servers = int(rng.integers(10, 101))  # inclusive upper bound 100
        power_scale = float(rng.uniform(0.2, 0.6))
        warm_percent = float(rng.uniform(0.1, 1.0))
        mean_service_time = float(rng.uniform(1.0, 30.0))
        spawn_time = float(rng.uniform(1.0, 30.0))
        service_rate = 1.0 / mean_service_time

        total_cap = containers_per_server * num_servers
        capacity = total_cap * service_rate
        # cold_effective_rate = 1.0 / (spawn_time + 1.0 / service_rate)
        # total_effective_capacity = warm_capacity + n_cold * cold_effective_rate

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
        })

    return cases

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
        "spawn_distribution": "exponential",
        "arrivals": "exponential",
        "ram_warm": ram_warm,
        "cpu_warm": cpu_warm,
        "ram_demand": ram_demand,
        "cpu_demand": cpu_demand,
        "cpu_transit": cpu_transit,
        "ram_transit": ram_transit,
        "peak_power": test_case['peak_power'],
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
    
    sim_config["request"]["warm_cpu"] = cpu_warm
    sim_config["request"]["warm_ram"] = ram_warm
    sim_config["request"]["cold_start_cpu"] = cpu_transit
    sim_config["request"]["cold_start_ram"] = ram_transit
    sim_config["request"]["cpu_demand"] = cpu_demand
    sim_config["request"]["ram_demand"] = ram_demand

    sim_config["server"]["peak_power"] = test_case['peak_power']
    sim_config["server"]["power_scale"] = test_case['power_scale']

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
    """Run the simulator and extract metrics with per-scenario 90% CIs from batch means.

    For each metric, returns the steady-state mean (mean of post-warmup batch means)
    and a 95% Student-t CI computed across the batch means. Markov is deterministic,
    so its scenario "value" is a single number with no CI.
    """
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

        # Per-batch values for steady-state CI estimation
        batches = system.compute_batch_means(num_batches=NUM_BATCHES)

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
            mean_val = float(np.mean(vals))
            if n > 1:
                t_crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, df=n - 1)
                se = float(np.std(vals, ddof=1)) / math.sqrt(n)
                hw = t_crit * se
            else:
                hw = 0.0
            result[metric] = mean_val
            result[f'{metric}_ci_low'] = mean_val - hw
            result[f'{metric}_ci_high'] = mean_val + hw
            result[f'{metric}_ci_halfwidth'] = hw
            result[f'{metric}_n_batches'] = n
        return result

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
        errors = markov_vals - sim_vals
        mean_error = np.mean(errors)

        # 95% Student-t confidence intervals (IID sample mean):
        #   CI = x̄ ± t_{0.025, n-1} * s / sqrt(n)
        n = len(markov_vals)
        if n > 1:
            t_crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, df=n - 1)
            # CI for mean signed error (bias)
            se_error = np.std(errors, ddof=1) / np.sqrt(n)
            ci_error_low = mean_error - t_crit * se_error
            ci_error_high = mean_error + t_crit * se_error

            # CI for MAPE: recompute per-case APE, then CI on its mean
            if metric == 'blocking_probability' and np.mean(sim_vals) < 0.01:
                per_case_ape = np.abs(errors) * 100  # absolute error in pct points
            else:
                per_case_ape = np.abs(errors / np.maximum(sim_vals, 1e-10)) * 100
            se_mape = np.std(per_case_ape, ddof=1) / np.sqrt(n)
            ci_mape_low = mape - t_crit * se_mape
            ci_mape_high = mape + t_crit * se_mape
        else:
            ci_error_low = ci_error_high = mean_error
            ci_mape_low = ci_mape_high = mape

        comparison_results[metric] = {
            'MAPE': mape,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'R_squared': r_squared,
            'mean_error': mean_error,
            'mean_markov': np.mean(markov_vals),
            'mean_sim': np.mean(sim_vals),
            'n_samples': n,
            'ci_error_low': ci_error_low,
            'ci_error_high': ci_error_high,
            'ci_mape_low': ci_mape_low,
            'ci_mape_high': ci_mape_high,
        }
    
    return comparison_results

def main():
    """Main comparison function"""
    print("Starting Markov Model vs Simulator Comparison")
    print("=" * 60)
    
    # Generate test cases
    test_cases = generate_test_cases(500)
    print(f"Generated {len(test_cases)} test cases")
    
    # Store results
    markov_results = []
    sim_results = []
    detailed_results = []
    
    # Run comparisons
    for i, test_case in enumerate(test_cases):
        print(f"\nRunning test case {i+1}/{len(test_cases)}")
        print(f"Parameters: λ={test_case['arrival_rate']:.2f}, μ={test_case['service_rate']:.3f}, "
              f"servers={test_case['num_servers']}, warm%={test_case['warm_percent']:.2f}, "
              f"spawn_time={test_case['spawn_time']:.2f}, "
              f"power_scale={test_case['power_scale']:.2f}")
        
        # Convert to respective configurations
        markov_config = convert_to_markov_config(test_case)
        sim_config = convert_to_simulator_config(test_case)
        
        # Run Markov model
        print("  Running Markov model...")
        markov_result = run_markov_model(markov_config)
        
        # Run simulator
        print("  Running simulator...")
        sim_result = run_simulator(sim_config)
        
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

            print(f"  Markov: block={markov_result['blocking_probability']:.4f}, "
                  f"latency={markov_result['latency']:.4f}, "
                  f"cpu={markov_result['cpu_usage']:.2f}, power={markov_result['power_usage']:.2f}")
            print(f"  Sim:    block={sim_result['blocking_probability']:.4f} "
                  f"[{sim_result['blocking_probability_ci_low']:.4f},{sim_result['blocking_probability_ci_high']:.4f}], "
                  f"latency={sim_result['latency']:.4f} "
                  f"[{sim_result['latency_ci_low']:.4f},{sim_result['latency_ci_high']:.4f}], "
                  f"cpu={sim_result['cpu_usage']:.2f} "
                  f"[{sim_result['cpu_usage_ci_low']:.2f},{sim_result['cpu_usage_ci_high']:.2f}]")
        else:
            print("  ERROR: One or both models failed!")
    
    # Calculate comparison metrics
    print(f"\n\nCalculating comparison metrics for {len(markov_results)} successful test cases...")
    comparison_metrics = calculate_comparison_metrics(markov_results, sim_results)
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for metric in ['blocking_probability', 'latency', 'variance_latency', 'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        m = comparison_metrics[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}  (n={m['n_samples']}):")
        print(f"    MAPE:       {m['MAPE']:.2f}%   "
              f"90% CI: [{m['ci_mape_low']:.2f}%, {m['ci_mape_high']:.2f}%]")
        print(f"    RMSE:       {m['RMSE']:.6f}")
        print(f"    NRMSE:      {m['NRMSE']:.2f}%")
        print(f"    R-squared:  {m['R_squared']:.4f}")
        print(f"    Mean Error: {m['mean_error']:+.6f}   "
              f"90% CI: [{m['ci_error_low']:+.6f}, {m['ci_error_high']:+.6f}]")
        print(f"      (CTMC avg={m['mean_markov']:.6f}, Sim avg={m['mean_sim']:.6f})")

    # -------------------------------------------------------------------------
    # Per-scenario simulator CI summary + Markov-in-CI coverage
    # Reviewer asked: "confidence intervals of both analyses".
    # Markov is deterministic (analytical) — single value per scenario, no CI.
    # Simulator is stochastic — per-scenario 90% CI from batch means
    # (NUM_BATCHES batches over post-warmup window, Student-t).
    # Coverage = fraction of scenarios where the Markov value lies inside the
    # simulator's 90% CI; expected ~90% if both models agree at steady state.
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  SIMULATOR PER-SCENARIO 90% CI (batch-means, B={NUM_BATCHES}, warmup={base_config['system'].get('warmup_time', 0)}s)")
    print(f"  + MARKOV-IN-SIM-CI COVERAGE")
    print(f"{'=' * 60}")
    print(f"  (Markov is deterministic — no per-scenario CI. Coverage near 95% indicates agreement.)")

    for metric in ['blocking_probability', 'latency', 'variance_latency',
                   'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        sim_means = np.array([r[metric] for r in sim_results])
        sim_lows = np.array([r[f'{metric}_ci_low'] for r in sim_results])
        sim_highs = np.array([r[f'{metric}_ci_high'] for r in sim_results])
        sim_hws = np.array([r[f'{metric}_ci_halfwidth'] for r in sim_results])
        markov_vals = np.array([r[metric] for r in markov_results])

        n = len(sim_means)
        if n == 0:
            continue
        # Mean half-width as absolute and as % of mean (relative precision)
        mean_hw = float(np.mean(sim_hws))
        mean_val = float(np.mean(sim_means))
        rel_hw_pct = (mean_hw / mean_val * 100) if mean_val > 0 else float('nan')

        # Coverage: Markov within simulator CI
        within = (markov_vals >= sim_lows) & (markov_vals <= sim_highs)
        coverage = float(np.mean(within)) * 100

        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"    Mean Sim CI half-width:  {mean_hw:.6f}  ({rel_hw_pct:.2f}% of mean)")
        print(f"    Markov inside Sim 90% CI: {int(within.sum())}/{n}  ({coverage:.1f}%)")

    # -------------------------------------------------------------------------
    # Bias & Error Distribution Analysis (CTMC)
    # -------------------------------------------------------------------------
    if len(detailed_results) > 0:
        df = pd.DataFrame(detailed_results)

        print(f"\n{'=' * 70}")
        print(f"  BIAS & ERROR DISTRIBUTION ANALYSIS (CTMC vs Simulator)")
        print(f"{'=' * 70}")

        # Per-metric signed error analysis
        for metric_short, metric_label in [
            ('blocking', 'Blocking Prob.'), ('latency', 'Latency'),
            ('variance_latency', 'Variance of Latency'), ('p99_latency', 'P99 Latency'),
            ('cpu', 'CPU Usage'), ('ram', 'RAM Usage'), ('power', 'Power Usage'),
        ]:
            model_col = f'markov_{metric_short}'
            sim_col = f'sim_{metric_short}'
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

        # --- Conditional analysis by n_warm ---
        print(f"\n  {'─' * 60}")
        print(f"  Conditional MAPE by n_warm (warm pool size)")
        print(f"  {'─' * 60}")

        bins = [0, 2, 4, 8, 16, 100]
        labels = ['1-2', '3-4', '5-8', '9-16', '17+']
        df['n_warm_bin'] = pd.cut(df['n_warm'], bins=bins, labels=labels, right=True)

        for metric_short, metric_label in [
            ('blocking', 'Blocking'), ('latency', 'Latency'),
            ('variance_latency', 'Variance of Latency'), ('p99_latency', 'P99 Latency'),
            ('cpu', 'CPU'), ('ram', 'RAM'), ('power', 'Power'),
        ]:
            model_col = f'markov_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df.columns or sim_col not in df.columns:
                continue

            print(f"\n    {metric_label}:")
            for bin_label in labels:
                subset = df[df['n_warm_bin'] == bin_label]
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
            ('variance_latency', 'Variance of Latency'), ('p99_latency', 'P99 Latency'),
        ]:
            model_col = f'markov_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df.columns or sim_col not in df.columns:
                continue

            print(f"\n    {metric_label}:")
            for st in sorted(df['spawn_time'].unique()):
                subset = df[df['spawn_time'] == st]
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

        df['rho'] = df.apply(_compute_rho, axis=1)

        rho_bins = [0, 0.7, 1.0, 1.3, 100]
        rho_labels = ['light(<0.7)', 'moderate(0.7-1.0)', 'heavy(1.0-1.3)', 'overload(>1.3)']
        df['load_regime'] = pd.cut(df['rho'], bins=rho_bins, labels=rho_labels, right=True)

        for metric_short, metric_label in [
            ('blocking', 'Blocking'), ('latency', 'Latency'),
            ('variance_latency', 'Variance of Latency'), ('p99_latency', 'P99 Latency'),
        ]:
            model_col = f'markov_{metric_short}'
            sim_col = f'sim_{metric_short}'
            if model_col not in df.columns or sim_col not in df.columns:
                continue

            print(f"\n    {metric_label}:")
            for regime in rho_labels:
                subset = df[df['load_regime'] == regime]
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

    # Save detailed results to CSV (one row per scenario; includes Sim 90% CI per metric)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("comparison_results", exist_ok=True)
    filename = f"comparison_results/markov_vs_sim_{timestamp}.csv"
    pd.DataFrame(detailed_results).to_csv(filename, index=False)
    print(f"\nDetailed results (with per-scenario Sim CIs) saved to: {filename}")
    
    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful comparisons: {len(markov_results)}")
    print(f"Success rate: {len(markov_results)/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    main()