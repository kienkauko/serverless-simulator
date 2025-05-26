#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script between Markov Model and Simulator
Runs both systems with the same input parameters and compares their results
"""

import numpy as np
import pandas as pd
import math
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
# from variables import config, request_stats, latency_stats
from Server import Server
from Request import Request
from Container import Container
from System import System

def generate_test_cases(num_cases=50):
    """Generate test cases by varying key parameters"""
    test_cases = []
    
    # Define parameter ranges
    arrival_rates = np.linspace(10, 30.0, 5)  # Lambda values
    service_rates = [0.25, 1.0, 4.0]  # Mu values
    num_servers_list = [5, 15, 30]
    warm_percents = [0.3, 0.5, 0.7, 1.0]
    spawn_times = [2, 4, 8]  # Container spawn times
    
    # Generate combinations
    case_id = 0
    for arrival_rate in arrival_rates:
        for service_rate in service_rates:
            for num_servers in num_servers_list:
                for warm_percent in warm_percents:
                    for spawn_time in spawn_times:
                        if case_id >= num_cases:
                            break
                        
                        test_cases.append({
                            'case_id': case_id,
                            'arrival_rate': arrival_rate,
                            'service_rate': service_rate,
                            'num_servers': num_servers,
                            'warm_percent': warm_percent,
                            'spawn_time': spawn_time
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
    
    return test_cases[:num_cases]

def convert_to_markov_config(test_case):
    """Convert test case to Markov model configuration"""
    
    # Fixed resource values (matching simulator defaults)
    cpu_warm = 1
    ram_warm = 30
    cpu_demand = 50
    ram_demand = 40
    
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
        "ram_demand": ram_demand,
        "cpu_demand": cpu_demand,
        "peak_power": 150.0,
        "power_scale": 0.5
    }
    
    return markov_config

def convert_to_simulator_config(test_case):
    """Convert test case to simulator configuration"""
    
    sim_config = {
        # System Parameters
        "system": {
            "num_servers": test_case['num_servers'],
            "sim_time": 9000,  # Longer simulation for stability
            "distribution": "deterministic",  
            "verbose": False,
            "warm_percent": test_case['warm_percent'],
        },
        
        # Server Parameters
        "server": {
            "cpu_capacity": 100.0,
            "ram_capacity": 100.0,
            "peak_power": 150.0,
            "power_scale": 0.5, 
        },
        
        # Request Parameters
        "request": {
            "arrival_rate": test_case['arrival_rate'],
            "service_rate": test_case['service_rate'],
            "warm_cpu": 1,
            "warm_ram": 30.0,
            "warm_cpu_model": 1,
            "warm_ram_model": 30.0,
            "cpu_demand": 50.0,
            "ram_demand": 40.0,
        },
        
        # Container Parameters
        "container": {
            "spawn_time": test_case['spawn_time'],
            "idle_cpu_timeout": 0,
            "idle_model_timeout": 0,
            "load_request_time": 0,
            "load_model_time": 0,
        },
        
        # Topology Parameters
        "topology": {
            "use_topology": False,
        }
    }
    
    return sim_config

def run_markov_model(markov_config):
    """Run the Markov model and extract metrics"""
    try:
        model = MarkovModel(markov_config, verbose=False)
        metrics = model.get_metrics()
        
        return {
            'blocking_probability': metrics['blocking_ratios'][0],
            'latency': metrics['latency'][0],
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
        system = System(env, sim_config, distribution=sim_config["system"]["distribution"], 
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
        if system.latency_stats['count'] > 0:
            avg_latency = system.latency_stats['total_latency'] / system.latency_stats['count']
        print(f"Total latency: {system.latency_stats['total_latency']:.10f}")
        print(f"Avg latency: {avg_latency:.10f}")
        mean_cpu_usage = system.get_mean_cpu_usage()
        mean_ram_usage = system.get_mean_ram_usage()
        mean_power_usage = system.get_mean_power_usage()

        return {
            'blocking_probability': blocking_probability,
            'latency': avg_latency,
            'cpu_usage': mean_cpu_usage,
            'ram_usage': mean_ram_usage,
            'power_usage': mean_power_usage
        }
        
    except Exception as e:
        print(f"Error in simulator: {e}")
        return None

def calculate_comparison_metrics(markov_results, sim_results):
    """Calculate MAPE, RMSE, and R-squared for comparison"""
    
    metrics = ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']
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
        
        comparison_results[metric] = {
            'MAPE': mape,
            'RMSE': rmse,
            'NRMSE': nrmse,  # Add this new metric
            'R_squared': r_squared
        }
    
    return comparison_results

def main():
    """Main comparison function"""
    print("Starting Markov Model vs Simulator Comparison")
    print("=" * 60)
    
    # Generate test cases
    test_cases = generate_test_cases(200)
    print(f"Generated {len(test_cases)} test cases")
    
    # Store results
    markov_results = []
    sim_results = []
    detailed_results = []
    
    # Run comparisons
    for i, test_case in enumerate(test_cases):
        print(f"\nRunning test case {i+1}/{len(test_cases)}")
        print(f"Parameters: λ={test_case['arrival_rate']:.1f}, μ={test_case['service_rate']:.1f}, "
              f"servers={test_case['num_servers']}, warm%={test_case['warm_percent']:.1f}, "
              f"spawn_time={test_case['spawn_time']}")
        
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
            
            # Store detailed results
            detailed_results.append({
                'case_id': test_case['case_id'],
                'arrival_rate': test_case['arrival_rate'],
                'service_rate': test_case['service_rate'],
                'num_servers': test_case['num_servers'],
                'warm_percent': test_case['warm_percent'],
                'spawn_time': test_case['spawn_time'],
                'markov_blocking': markov_result['blocking_probability'],
                'sim_blocking': sim_result['blocking_probability'],
                'markov_latency': markov_result['latency'],
                'sim_latency': sim_result['latency'],
                'markov_cpu': markov_result['cpu_usage'],
                'sim_cpu': sim_result['cpu_usage'],
                'markov_ram': markov_result['ram_usage'],
                'sim_ram': sim_result['ram_usage']
            })
            
            print(f"  Markov: block={markov_result['blocking_probability']:.4f}, "
                  f"latency={markov_result['latency']:.4f}, "
                  f"cpu={markov_result['cpu_usage']:.2f}, ram={markov_result['ram_usage']:.2f},"
                  f"power={markov_result['power_usage']:.2f}")
            print(f"  Sim:    block={sim_result['blocking_probability']:.4f}, "
                  f"latency={sim_result['latency']:.4f}, "
                  f"cpu={sim_result['cpu_usage']:.2f}, ram={sim_result['ram_usage']:.2f}",
                  f"power={sim_result['power_usage']:.2f}")
        else:
            print("  ERROR: One or both models failed!")
    
    # Calculate comparison metrics
    print(f"\n\nCalculating comparison metrics for {len(markov_results)} successful test cases...")
    comparison_metrics = calculate_comparison_metrics(markov_results, sim_results)
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for metric in ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  MAPE:      {comparison_metrics[metric]['MAPE']:.2f}%")
        print(f"  RMSE:      {comparison_metrics[metric]['RMSE']:.6f}")
        print(f"  R-squared: {comparison_metrics[metric]['R_squared']:.4f}")
    
    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results/markov_vs_sim_{timestamp}.csv"
    
    # Create directory if it doesn't exist
    os.makedirs("comparison_results", exist_ok=True)
    
    # Create a combined dataset with model identifier
    combined_results = []
    
    # Add Markov results
    for i, (test_case, markov_result) in enumerate(zip(test_cases[:len(markov_results)], markov_results)):
        combined_results.append({
            'case_id': test_case['case_id'],
            'model_type': 'Markov',
            'arrival_rate': test_case['arrival_rate'],
            'service_rate': test_case['service_rate'],
            'num_servers': test_case['num_servers'],
            'warm_percent': test_case['warm_percent'],
            'spawn_time': test_case['spawn_time'],
            'blocking_probability': markov_result['blocking_probability'],
            'latency': markov_result['latency'],
            'cpu_usage': markov_result['cpu_usage'],
            'ram_usage': markov_result['ram_usage']
        })
    
    # Add Simulator results
    for i, (test_case, sim_result) in enumerate(zip(test_cases[:len(sim_results)], sim_results)):
        combined_results.append({
            'case_id': test_case['case_id'],
            'model_type': 'Simulator',
            'arrival_rate': test_case['arrival_rate'],
            'service_rate': test_case['service_rate'],
            'num_servers': test_case['num_servers'],
            'warm_percent': test_case['warm_percent'],
            'spawn_time': test_case['spawn_time'],
            'blocking_probability': sim_result['blocking_probability'],
            'latency': sim_result['latency'],
            'cpu_usage': sim_result['cpu_usage'],
            'ram_usage': sim_result['ram_usage']
        })
    
    # Save combined results
    df_combined = pd.DataFrame(combined_results)
    df_combined.to_csv(filename, index=False)
    print(f"\nCombined results saved to: {filename}")
    
    df = pd.DataFrame(detailed_results)
    # df.to_csv(filename, index=False)
    # print(f"\nDetailed results saved to: {filename}")
    
    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful comparisons: {len(markov_results)}")
    print(f"Success rate: {len(markov_results)/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    main()