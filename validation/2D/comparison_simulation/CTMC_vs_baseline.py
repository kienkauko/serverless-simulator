#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script: 2D CTMC Markov Model vs Cascaded Erlang B Baseline
Runs both models with the same input parameters and compares their results.

The cascaded Erlang B baseline treats the warm pool and cold pool as two
independent M/M/n/n loss systems in series, which is an approximation
because it ignores the state-space coupling between the two pools.
"""

import numpy as np
import pandas as pd
import math
import os
from datetime import datetime
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# Import 2D CTMC Markov model
from Markov.model_2D import MarkovModel


# =============================================================================
# Cascaded Erlang B Baseline Model
# =============================================================================

def erlang_b(offered_load, num_servers):
    """
    Compute the Erlang B blocking probability.
    
    B(A, n) = (A^n / n!) / sum_{k=0}^{n} (A^k / k!)
    
    Uses iterative (Jagerman's) formula for numerical stability:
        B(A, 0) = 1
        B(A, k) = (A * B(A, k-1)) / (k + A * B(A, k-1))
    
    Args:
        offered_load: A = lambda * E[B], total offered load in Erlangs
        num_servers: n, number of servers (loss system capacity)
    
    Returns:
        Blocking probability
    """
    if num_servers == 0:
        return 1.0
    if offered_load <= 0:
        return 0.0
    
    # Jagerman's recursive formula for numerical stability
    inv_b = 1.0
    for k in range(1, num_servers + 1):
        inv_b = 1.0 + (k / offered_load) * inv_b
    return 1.0 / inv_b


class CascadedErlangBModel:
    """
    Cascaded Erlang B baseline model.
    
    Models the warm pool and cold pool as two independent M/M/n/n loss systems.
    Overflow from the warm pool feeds into the cold pool as a Poisson process
    (which is the key approximation error — overflow is actually non-Poisson).
    """
    
    def __init__(self, config):
        self._lam = config["lam"]
        self._mu = config["mu"]
        self._spawn_rate = config["spawn_rate"]
        self._n_warm = config["queue_warm"]
        self._n_cold = config["queue_cold"]
        self._cpu_warm = config["cpu_warm"]
        self._cpu_active = config["cpu_demand"]
        self._ram_warm = config["ram_warm"]
        self._ram_active = config["ram_demand"]
        self._peak_power = config["peak_power"]
        self._power_scale = config["power_scale"]
        
        # Service times
        self._E_Bw = 1.0 / self._mu                           # Warm service time
        self._E_Bc = 1.0 / self._spawn_rate                   # Cold start (spawn) time
        self._E_B_cold_total = self._E_Bc + self._E_Bw        # Total cold processing time
        
        # Compute metrics
        self._compute_all()
    
    def _compute_all(self):
        """Compute all cascaded Erlang B metrics."""
        
        # --- Warm pool: M/M/n_w/n_w ---
        self._A_warm = self._lam * self._E_Bw                  # Offered load to warm pool
        self._pb_warm = erlang_b(self._A_warm, self._n_warm)    # Warm blocking probability
        self._lam_warm = self._lam * (1 - self._pb_warm)        # Arrival rate served by warm pool
        
        # --- Cold pool: M/M/n_c/n_c with overflow traffic (assumed Poisson) ---
        self._lam_overflow = self._lam * self._pb_warm          # Overflow rate from warm pool
        self._A_cold = self._lam_overflow * self._E_B_cold_total  # Offered load to cold pool
        self._pb_cold = erlang_b(self._A_cold, self._n_cold)    # Cold blocking probability
        self._lam_cold = self._lam_overflow * (1 - self._pb_cold)  # Arrival rate served by cold pool
        
        # --- Overall system metrics ---
        self._pb = self._pb_warm * self._pb_cold                # Overall blocking (product form)
        self._lam_a = self._lam_warm + self._lam_cold           # Total successful arrival rate
        
        # Mean number of busy servers in each pool (Erlang B property: E[X] = A*(1-B))
        self._E_Xw = self._A_warm * (1 - self._pb_warm)        # Mean busy warm servers
        self._E_Xc = self._A_cold * (1 - self._pb_cold)        # Mean busy cold servers
        self._E_X = self._E_Xw + self._E_Xc                    # Mean total jobs in system
    
    def get_blocking_probability(self):
        return self._pb
    
    def get_latency(self):
        """Compute mean response time via Little's Law: E[B_r] = E[X] / lambda_a"""
        if self._lam_a <= 0:
            return 0.0
        return self._E_X / self._lam_a
    
    def get_resource_usage(self, resource):
        """
        Compute resource usage matching the Markov model's formula.
        
        r = n_w * r_warm + E[X_w] * (r_active - r_warm) + E[X_c] * r_cold_mean
        
        where r_cold_mean is the time-weighted average resource during cold processing.
        """
        if resource == "cpu":
            active = self._cpu_active
            warm = self._cpu_warm
            transit = 3.22  # cold start CPU (matching Markov model)
        elif resource == "ram":
            active = self._ram_active
            warm = self._ram_warm
            transit = warm   # cold start RAM = warm RAM (matching Markov model)
        else:
            raise ValueError("Resource must be either 'cpu' or 'ram'")
        
        spawn_time = 1.0 / self._spawn_rate
        serving_time = 1.0 / self._mu
        mean_cold_consume = (spawn_time * transit + serving_time * active) / (spawn_time + serving_time)
        
        resource_usage = (self._n_warm * warm 
                         + self._E_Xw * (active - warm)
                         + self._E_Xc * mean_cold_consume)
        
        return resource_usage
    
    def get_power_usage(self, cpu_usage):
        """Compute power usage matching the Markov model's formula."""
        driven_resource = max(self._cpu_active, self._ram_active)
        num_con_per_server = math.floor(100 / driven_resource)
        num_job = self._n_warm + self._E_Xc
        on_server = math.ceil(num_job / num_con_per_server)
        base_power = self._peak_power * self._power_scale
        
        return on_server * base_power + (cpu_usage / 100) * (self._peak_power - base_power)
    
    def get_metrics(self):
        """Return metrics in the same format as MarkovModel.get_metrics()."""
        cpu_usage = self.get_resource_usage("cpu")
        ram_usage = self.get_resource_usage("ram")
        power_usage = self.get_power_usage(cpu_usage)
        
        return {
            'blocking_probability': self._pb,
            'blocking_warm': self._pb_warm,
            'blocking_cold': self._pb_cold,
            'latency': self.get_latency(),
            'cpu_usage': cpu_usage,
            'ram_usage': ram_usage,
            'power_usage': power_usage,
            'E_Xw': self._E_Xw,
            'E_Xc': self._E_Xc,
        }


# =============================================================================
# Test Case Generation & Config Conversion
# =============================================================================

# Fixed resource values (matching simulator defaults)
cpu_warm = 1
ram_warm = 30
cpu_demand = 50
ram_demand = 40

def generate_test_cases(num_cases=50):
    """Generate test cases specifically designed to stress the independence
    assumption of the cascaded Erlang B model.

    The Erlang B decomposition fails most when:
    1. The warm pool is frequently saturated (high overflow)
    2. The cold pool is moderately loaded (overflow burstiness matters)
    3. Cold start time is large relative to service time (amplifies coupling)
    4. The warm pool is small (overflow is more bursty from small pools)

    We target the "interesting" regime where offered load ~ system capacity.
    """
    test_cases = []

    max_resource = max(cpu_demand, ram_demand)
    containers_per_server = math.floor(100 / max_resource)  # = 2

    case_id = 0

    # For each (num_servers, warm_percent, spawn_time),
    # compute the capacity and sweep lambda around it
    num_servers_list = [3, 5, 8, 10, 15, 20]
    warm_percents = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    spawn_times = [2, 4, 6, 8, 12]
    # Use moderate service rates so cold start overhead is significant
    service_rates = [0.5, 1.0, 2.0]

    for num_servers in num_servers_list:
        total_cap = containers_per_server * num_servers
        for service_rate in service_rates:
            for warm_percent in warm_percents:
                n_warm = int(total_cap * warm_percent)
                n_cold = total_cap - n_warm

                if n_warm < 1 or n_cold < 1:
                    continue

                # The warm pool saturates at lambda ~ n_warm * mu
                warm_capacity = n_warm * service_rate

                for spawn_time in spawn_times:
                    cold_effective_rate = 1.0 / (spawn_time + 1.0 / service_rate)
                    total_effective_capacity = warm_capacity + n_cold * cold_effective_rate

                    # Sweep lambda from just above warm capacity to around total capacity
                    load_fractions = [0.7, 0.85, 1.0, 1.1, 1.2, 1.4]

                    for frac in load_fractions:
                        arrival_rate = frac * total_effective_capacity
                        if arrival_rate < 0.5:
                            continue

                        # Ensure warm pool is actually stressed
                        rho_warm = arrival_rate / service_rate
                        if rho_warm < n_warm * 0.5:
                            continue

                        test_cases.append({
                            'case_id': case_id,
                            'arrival_rate': round(arrival_rate, 2),
                            'service_rate': service_rate,
                            'num_servers': num_servers,
                            'warm_percent': warm_percent,
                            'spawn_time': spawn_time,
                        })
                        case_id += 1

                        if case_id >= num_cases:
                            break
                    if case_id >= num_cases:
                        break
                if case_id >= num_cases:
                    break
            if case_id >= num_cases:
                break
        if case_id >= num_cases:
            break

    # Add extreme coupling cases: small warm pool + large cold start
    extreme_cases = [
        {'arrival_rate': 3.0, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.2, 'spawn_time': 10},
        {'arrival_rate': 4.0, 'service_rate': 1.0, 'num_servers': 3,
         'warm_percent': 0.3, 'spawn_time': 8},
        {'arrival_rate': 5.0, 'service_rate': 0.5, 'num_servers': 5,
         'warm_percent': 0.2, 'spawn_time': 12},
        {'arrival_rate': 8.0, 'service_rate': 1.0, 'num_servers': 5,
         'warm_percent': 0.4, 'spawn_time': 15},
        {'arrival_rate': 10.0, 'service_rate': 2.0, 'num_servers': 5,
         'warm_percent': 0.3, 'spawn_time': 10},
        {'arrival_rate': 15.0, 'service_rate': 1.0, 'num_servers': 10,
         'warm_percent': 0.3, 'spawn_time': 8},
        {'arrival_rate': 20.0, 'service_rate': 1.0, 'num_servers': 10,
         'warm_percent': 0.5, 'spawn_time': 6},
        {'arrival_rate': 25.0, 'service_rate': 2.0, 'num_servers': 15,
         'warm_percent': 0.2, 'spawn_time': 10},
    ]

    for ec in extreme_cases:
        if case_id >= num_cases:
            break
        ec['case_id'] = case_id
        test_cases.append(ec)
        case_id += 1

    return test_cases[:num_cases]


def convert_to_model_config(test_case):
    """Convert test case to model configuration (shared by both models)."""
    max_resource = max(cpu_demand, ram_demand)
    total_containers = math.floor(100 / max_resource) * test_case['num_servers']
    queue_warm = int(total_containers * test_case['warm_percent'])
    queue_cold = total_containers - queue_warm
    
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


# =============================================================================
# Model Runners
# =============================================================================

def run_ctmc_model(config):
    """Run the 2D CTMC Markov model and extract metrics."""
    try:
        model = MarkovModel(config, verbose=False)
        metrics = model.get_metrics()
        return {
            'blocking_probability': metrics['blocking_ratios'][0],
            'latency': metrics['latency'][0],
            'cpu_usage': metrics['cpu_usage'][0],
            'ram_usage': metrics['ram_usage'][0],
            'power_usage': metrics['power_usage'][0],
        }
    except Exception as e:
        print(f"  Error in CTMC model: {e}")
        return None


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


# =============================================================================
# Comparison Metrics
# =============================================================================

def calculate_comparison_metrics(ctmc_results, erlang_results):
    """
    Calculate MAPE, RMSE, NRMSE, and R-squared.
    CTMC is treated as the reference (ground truth).
    """
    metrics = ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']
    comparison_results = {}
    
    for metric in metrics:
        ctmc_vals = np.array([r[metric] for r in ctmc_results if r is not None])
        erlang_vals = np.array([r[metric] for r in erlang_results if r is not None])
        
        min_len = min(len(ctmc_vals), len(erlang_vals))
        ctmc_vals = ctmc_vals[:min_len]
        erlang_vals = erlang_vals[:min_len]
        
        if len(ctmc_vals) == 0:
            comparison_results[metric] = {
                'MAPE': float('inf'), 'RMSE': float('inf'),
                'NRMSE': float('inf'), 'R_squared': 0.0,
                'mean_ctmc': 0.0, 'mean_erlang': 0.0,
            }
            continue
        
        # MAPE (Erlang B vs CTMC reference)
        if metric == 'blocking_probability':
            threshold = 0.01
            if np.mean(ctmc_vals) < threshold:
                mape = np.mean(np.abs(erlang_vals - ctmc_vals)) * 100
            else:
                mape = np.mean(np.abs((erlang_vals - ctmc_vals) / np.maximum(ctmc_vals, 1e-10))) * 100
        else:
            mape = np.mean(np.abs((erlang_vals - ctmc_vals) / np.maximum(ctmc_vals, 1e-10))) * 100
        
        # RMSE
        rmse = np.sqrt(np.mean((erlang_vals - ctmc_vals) ** 2))
        
        # Normalized RMSE
        nrmse = (rmse / np.mean(ctmc_vals)) * 100 if np.mean(ctmc_vals) > 0 else float('inf')
        
        # R-squared (using CTMC as reference)
        ss_res = np.sum((erlang_vals - ctmc_vals) ** 2)
        ss_tot = np.sum((ctmc_vals - np.mean(ctmc_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Signed mean error (bias): positive means Erlang B overestimates
        mean_error = np.mean(erlang_vals - ctmc_vals)
        
        comparison_results[metric] = {
            'MAPE': mape,
            'RMSE': rmse,
            'NRMSE': nrmse,
            'R_squared': r_squared,
            'mean_error': mean_error,
            'mean_ctmc': np.mean(ctmc_vals),
            'mean_erlang': np.mean(erlang_vals),
        }
    
    return comparison_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  2D CTMC Markov Model  vs  Cascaded Erlang B Baseline")
    print("=" * 70)
    
    # Generate test cases
    test_cases = generate_test_cases(200)
    print(f"Generated {len(test_cases)} test cases\n")
    
    ctmc_results = []
    erlang_results = []
    detailed_results = []
    
    for i, tc in enumerate(test_cases):
        print(f"Test case {i+1}/{len(test_cases)}  |  "
              f"λ={tc['arrival_rate']:.1f}, μ={tc['service_rate']:.2f}, "
              f"servers={tc['num_servers']}, warm%={tc['warm_percent']:.1f}, "
              f"spawn_t={tc['spawn_time']}")
        
        config = convert_to_model_config(tc)
        print(f"  n_warm={config['queue_warm']}, n_cold={config['queue_cold']}")
        
        # Run both models with the same config
        ctmc_result = run_ctmc_model(config)
        erlang_result = run_erlang_b_model(config)
        
        if ctmc_result is not None and erlang_result is not None:
            ctmc_results.append(ctmc_result)
            erlang_results.append(erlang_result)
            
            detailed_results.append({
                'case_id': tc['case_id'],
                'arrival_rate': tc['arrival_rate'],
                'service_rate': tc['service_rate'],
                'num_servers': tc['num_servers'],
                'warm_percent': tc['warm_percent'],
                'spawn_time': tc['spawn_time'],
                'n_warm': config['queue_warm'],
                'n_cold': config['queue_cold'],
                # CTMC
                'ctmc_blocking': ctmc_result['blocking_probability'],
                'ctmc_latency': ctmc_result['latency'],
                'ctmc_cpu': ctmc_result['cpu_usage'],
                'ctmc_ram': ctmc_result['ram_usage'],
                'ctmc_power': ctmc_result['power_usage'],
                # Erlang B
                'erlang_blocking': erlang_result['blocking_probability'],
                'erlang_latency': erlang_result['latency'],
                'erlang_cpu': erlang_result['cpu_usage'],
                'erlang_ram': erlang_result['ram_usage'],
                'erlang_power': erlang_result['power_usage'],
            })
            
            print(f"  CTMC:     block={ctmc_result['blocking_probability']:.6f}, "
                  f"latency={ctmc_result['latency']:.4f}, "
                  f"cpu={ctmc_result['cpu_usage']:.2f}, ram={ctmc_result['ram_usage']:.2f}, "
                  f"power={ctmc_result['power_usage']:.2f}")
            print(f"  ErlangB:  block={erlang_result['blocking_probability']:.6f}, "
                  f"latency={erlang_result['latency']:.4f}, "
                  f"cpu={erlang_result['cpu_usage']:.2f}, ram={erlang_result['ram_usage']:.2f}, "
                  f"power={erlang_result['power_usage']:.2f}")
        else:
            print("  ERROR: One or both models failed!")
    
    # -------------------------------------------------------------------------
    # Comparison Metrics
    # -------------------------------------------------------------------------
    print(f"\n\nCalculating comparison metrics for {len(ctmc_results)} successful test cases...")
    comparison = calculate_comparison_metrics(ctmc_results, erlang_results)
    
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS  (Erlang B deviation from CTMC reference)")
    print("=" * 70)
    
    for metric in ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        m = comparison[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"    MAPE:       {m['MAPE']:.2f}%")
        print(f"    RMSE:       {m['RMSE']:.6f}")
        print(f"    NRMSE:      {m['NRMSE']:.2f}%")
        print(f"    R-squared:  {m['R_squared']:.4f}")
        print(f"    Mean Error: {m['mean_error']:+.6f}  "
              f"(CTMC avg={m['mean_ctmc']:.6f}, ErlangB avg={m['mean_erlang']:.6f})")
    
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    # os.makedirs("comparison_results", exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # # Detailed per-case results
    # filename_detail = f"comparison_results/ctmc_vs_erlangb_{timestamp}.csv"
    # df_detail = pd.DataFrame(detailed_results)
    # df_detail.to_csv(filename_detail, index=False)
    # print(f"\nDetailed results saved to: {filename_detail}")
    
    # # Combined long-format results (for easier plotting)
    # combined = []
    # for i, (tc, cr, er) in enumerate(zip(
    #         test_cases[:len(ctmc_results)], ctmc_results, erlang_results)):
    #     base = {
    #         'case_id': tc['case_id'],
    #         'arrival_rate': tc['arrival_rate'],
    #         'service_rate': tc['service_rate'],
    #         'num_servers': tc['num_servers'],
    #         'warm_percent': tc['warm_percent'],
    #         'spawn_time': tc['spawn_time'],
    #     }
    #     combined.append({**base, 'model_type': 'CTMC', **cr})
    #     combined.append({**base, 'model_type': 'ErlangB', **er})
    
    # filename_combined = f"comparison_results/ctmc_vs_erlangb_combined_{timestamp}.csv"
    # pd.DataFrame(combined).to_csv(filename_combined, index=False)
    # print(f"Combined results saved to: {filename_combined}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    # print(f"\n{'=' * 70}")
    # print(f"  SUMMARY")
    # print(f"{'=' * 70}")
    # print(f"  Total test cases:          {len(test_cases)}")
    # print(f"  Successful comparisons:    {len(ctmc_results)}")
    # print(f"  Success rate:              {len(ctmc_results)/len(test_cases)*100:.1f}%")
    # print(f"\n  Key insight: The cascaded Erlang B model treats the warm and cold")
    # print(f"  pools as independent, ignoring that overflow traffic from the warm")
    # print(f"  pool is bursty (non-Poisson). The 2D CTMC captures the exact joint")
    # print(f"  state distribution π(i,j), yielding more accurate results.")


if __name__ == "__main__":
    main()
