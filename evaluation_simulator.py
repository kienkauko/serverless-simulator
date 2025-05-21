#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the serverless simulator with varying timeout rate (theta)
This script analyzes how different values of theta affect various metrics
across multiple system configurations
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from scipy import stats
import pandas as pd
import itertools
import simpy
import random
import copy
import math
# Import simulator components
from System import System
from Server import Server
from variables import config as base_config
from variables import request_stats, latency_stats

def run_simulation(sim_config, reset_stats=True, traffic_pattern='poisson'):
    """
    Run a single simulation with the provided configuration
    
    Parameters:
    -----------
    sim_config : dict
        Dictionary containing simulator configuration
    reset_stats : bool
        Whether to reset global statistics before the simulation
    traffic_pattern : str
        Traffic pattern type: 'poisson' or 'day_night'
        
    Returns:
    --------
    metrics : dict
        Dictionary containing the simulation metrics
    """
    # Reset global statistics if requested
    if reset_stats:
        # Reset request_stats
        for key in request_stats:
            request_stats[key] = 0
            
        # Reset latency_stats
        for key in latency_stats:
            latency_stats[key] = 0
    
    # Create the simulation environment
    env = simpy.Environment()
    
    # Create the System with config dictionary
    system = System(env, sim_config, 
                    distribution=sim_config["system"]["distribution"], 
                    pattern_type=traffic_pattern,
                    verbose=sim_config["system"]["verbose"])
    
    # Add servers directly to the system
    for i in range(sim_config["system"]["num_servers"]):
        server = Server(env, f"Server-{i}", 
                       sim_config["server"]["cpu_capacity"], 
                       sim_config["server"]["ram_capacity"])
        system.add_server(server)
    
    # Start the request generation process
    env.process(system.request_generator())
    
    # Run the simulation
    sim_time = sim_config["system"]["sim_time"]
    env.run(until=sim_time)
    
    # Calculate metrics
    blocking_prob = 0
    if request_stats['generated'] > 0:
        blocking_prob = request_stats['blocked_no_server_capacity'] / request_stats['generated']
    
    # Calculate average latencies
    avg_waiting_time = 0
    if latency_stats['count'] > 0:
        avg_waiting_time = latency_stats['waiting_time'] / latency_stats['count']
    
    # Get resource usage
    mean_cpu_usage = system.get_mean_cpu_usage() / system.get_mean_requests_in_system()
    mean_ram_usage = system.get_mean_ram_usage() / system.get_mean_requests_in_system()
    
    # Calculate Little's Law metrics
    avg_waiting_count = 0
    effective_arrival_rate = 0
    little_law_wait_time = 0
    
    if env.now > 0:
        avg_waiting_count = system.total_waiting_area / env.now
        effective_arrival_rate = request_stats['processed'] / env.now
        little_law_wait_time = avg_waiting_count / effective_arrival_rate if effective_arrival_rate > 0 else 0
    
    # Compile metrics
    metrics = {
        "blocking_prob": blocking_prob,
        "waiting_time": avg_waiting_time,
        "cpu_usage": mean_cpu_usage,
        "ram_usage": mean_ram_usage,
        "effective_arrival_rate": effective_arrival_rate,
        "waiting_requests": avg_waiting_count,
        "processing_requests": request_stats['processed'] / sim_time,
        "request_stats": copy.deepcopy(request_stats),
        "latency_stats": copy.deepcopy(latency_stats)
    }
    
    return metrics

def find_best_theta(metrics_dict, weights=None):
    """
    Find the best theta value that optimizes multiple metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing theta_values and metrics arrays
    weights : dict, optional
        Dictionary of weights for each metric. Default weights prioritize
        blocking probability and waiting time over resource usage.
    
    Returns:
    --------
    best_theta : float
        The theta value that yields the optimal combined performance
    best_metrics : dict
        Dictionary of metrics at the best theta value
    ranking : dict
        Dictionary containing normalized scores and overall ranking
    """
    if weights is None:
        # Default weights if none provided (can be adjusted based on priorities)
        weights = {
            'blocking_probs': 0.4,    # Higher weight for blocking probability
            'waiting_times': 0.3,     # High weight for waiting time
            'cpu_usages': 0.15,       # Lower weight for CPU usage
            'ram_usages': 0.15        # Lower weight for RAM usage
        }
    
    # Extract data
    theta_values = metrics_dict['theta_values']
    blocking_probs = np.array(metrics_dict['blocking_probs'])
    waiting_times = np.array(metrics_dict['waiting_times'])
    cpu_usages = np.array(metrics_dict['cpu_usages'])
    ram_usages = np.array(metrics_dict['ram_usages'])
    
    # Normalize each metric to [0,1] range
    # For all metrics, lower is better, so we normalize linearly
    norm_blocking = (blocking_probs - np.min(blocking_probs)) / (np.max(blocking_probs) - np.min(blocking_probs) + 1e-10)
    norm_waiting = (waiting_times - np.min(waiting_times)) / (np.max(waiting_times) - np.min(waiting_times) + 1e-10)
    norm_cpu = (cpu_usages - np.min(cpu_usages)) / (np.max(cpu_usages) - np.min(cpu_usages) + 1e-10)
    norm_ram = (ram_usages - np.min(ram_usages)) / (np.max(ram_usages) - np.min(ram_usages) + 1e-10)
    
    # Calculate weighted score (lower is better)
    weighted_scores = (
        weights['blocking_probs'] * norm_blocking +
        weights['waiting_times'] * norm_waiting +
        weights['cpu_usages'] * norm_cpu +
        weights['ram_usages'] * norm_ram
    )
    
    # Find the index of the minimum score
    best_idx = np.argmin(weighted_scores)
    best_theta = theta_values[best_idx]
    
    # Create results dictionary
    best_metrics = {
        'theta': best_theta,
        'blocking_prob': blocking_probs[best_idx],
        'waiting_time': waiting_times[best_idx],
        'cpu_usage': cpu_usages[best_idx],
        'ram_usage': ram_usages[best_idx],
        'weighted_score': weighted_scores[best_idx]
    }
    
    # Create ranking dictionary with normalized scores
    ranking = {
        'theta_values': theta_values,
        'norm_blocking': norm_blocking,
        'norm_waiting': norm_waiting,
        'norm_cpu': norm_cpu,
        'norm_ram': norm_ram,
        'weighted_scores': weighted_scores
    }
    
    return best_theta, best_metrics, ranking

def plot_optimization_results(theta_values, ranking, best_theta, plots_dir=None, timestamp=None):
    """
    Plot the optimization results and highlight the best theta value
    
    Parameters:
    -----------
    theta_values : array
        Array of theta values that were evaluated
    ranking : dict
        Dictionary with normalized scores for each metric
    best_theta : float
        The best theta value identified
    plots_dir : str, optional
        Directory to save plots
    timestamp : str, optional
        Timestamp to use in filenames
    """
    # Define figure style variables for consistent visualization
    fig_style = {
        "figure_size": (12, 8),            # Figure size in inches
        "title_fontsize": 14,              # Title font size
        "label_fontsize": 16,              # Axis label font size
        "tick_fontsize": 16,               # Tick label font size
        "legend_fontsize": 14,             # Legend font size
        "line_width": 3,                   # Line width for plots
        "marker_size": 6,                  # Marker size
        "grid_alpha": 0.3,                 # Grid transparency
        "marker_style": 'o',               # Marker style
        "dpi": 300                         # DPI for saved figures
    }
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if plots_dir is None:
        plots_dir = "theta_evaluation_plots"
        os.makedirs(plots_dir, exist_ok=True)
    
    # Find the index of the best theta in the array
    best_idx = np.abs(theta_values - best_theta).argmin()
    
    # Create composite plot showing all normalized metrics
    plt.figure(figsize=fig_style["figure_size"])
    plt.plot(theta_values, ranking['norm_blocking'], 'b-', linewidth=fig_style["line_width"], label='Blocking Probability')
    plt.plot(theta_values, ranking['norm_waiting'], 'r-', linewidth=fig_style["line_width"], label='Waiting Time')
    plt.plot(theta_values, ranking['norm_cpu'], 'g-', linewidth=fig_style["line_width"], label='CPU Usage')
    plt.plot(theta_values, ranking['norm_ram'], 'm-', linewidth=fig_style["line_width"], label='RAM Usage')
    plt.plot(theta_values, ranking['weighted_scores'], 'k--', linewidth=fig_style["line_width"], label='Weighted Score')
    
    # Mark the best theta
    plt.axvline(x=best_theta, color='orange', linestyle='-', alpha=0.5)
    plt.plot(best_theta, ranking['weighted_scores'][best_idx], 'ro', markersize=10, 
             label=f'Best θ = {best_theta:.4f}')
    
    plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Normalized Score (lower is better)', fontsize=fig_style["label_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.savefig(f"{plots_dir}/optimization_results_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()

def evaluate_multiple_configs(n_iterations=1, traffic_pattern='poisson'):
    """
    Evaluates the simulator across multiple system configurations and iterations
    
    Parameters:
    -----------
    n_iterations : int
        Number of iterations to run for each configuration to calculate confidence intervals
    traffic_pattern : str
        Traffic pattern type: 'poisson' or 'day_night'
    
    Returns:
    --------
    all_results : DataFrame
        Pandas DataFrame containing all evaluation results
    """
    # Define parameter ranges to test
    param_ranges = {
        "mu": np.linspace(1.0, 10.0, 5),                # Service rate range from 1 to 10
        "alpha": np.linspace(0.1, 5.0, 5),              # Request activation rate range from 0.1 to 3.0
        "max_queue": np.linspace(5, 20, 5, dtype=int)   # Maximum queue size range from 5 to 20
    }
    
    # For day_night traffic pattern, lam is fixed since the pattern itself varies the arrival rate
    # For poisson traffic pattern, we vary lam as in the original code
    if traffic_pattern == 'poisson':
        param_ranges["lam"] = np.linspace(1.0, 10.0, 5)  # Arrival rate range from 1 to 10
    else:  # day_night pattern
        param_ranges["lam"] = np.array([5.0])  # Fixed base arrival rate
        print("Using day_night traffic pattern with fixed base arrival rate of 5.0")
    
    # Create directory for results if it doesn't exist
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for result filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a list to store all results
    all_results = []
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        param_ranges["lam"],
        param_ranges["mu"],
        param_ranges["alpha"],
        param_ranges["max_queue"]
    ))
    
    print(f"Evaluating {len(param_combinations)} parameter combinations with {n_iterations} iterations each...")
    
    # For each parameter combination
    for combo_idx, (lam, mu, alpha, max_queue) in enumerate(param_combinations):
        print(f"Configuration {combo_idx+1}/{len(param_combinations)}: lam={lam}, mu={mu}, alpha={alpha}, max_queue={max_queue}")
        
        # Define theta values to test for this configuration
        theta_values = np.linspace(0.01, 1.0, 5)
        
        # For each theta value
        for theta_idx, theta in enumerate(theta_values):
            print(f"  Processing theta = {theta:.2f} ({theta_idx+1}/{len(theta_values)})")
            
            # Run multiple iterations to calculate confidence intervals
            iteration_results = []
            for iteration in range(n_iterations):
                # Create configuration for this run
                sim_config = copy.deepcopy(base_config)
                
                # Set parameters for this configuration
                # Set parameters for request characteristics
                sim_config["request"]["ram_warm"] = 10
                sim_config["request"]["cpu_warm"] = 1
                sim_config["request"]["ram_demand"] = 25
                sim_config["request"]["cpu_demand"] = 30
                # Store these values for easier access
                ram_demand = sim_config["request"]["ram_demand"]
                cpu_demand = sim_config["request"]["cpu_demand"]
                sim_config["system"]["num_servers"] = math.ceil(max_queue/math.floor(100/max(cpu_demand, ram_demand)))

                sim_config["system"]["sim_time"] = 500
                sim_config["system"]["verbose"] = False
                sim_config["request"]["arrival_rate"] = lam
                sim_config["request"]["service_rate"] = mu
                sim_config["container"]["spawn_time"] = 1/alpha  # alpha is spawn rate, so spawn_time = 1/alpha
                sim_config["container"]["idle_timeout"] = 1/theta  # theta is timeout rate, so idle_timeout = 1/theta
                sim_config["system"]["distribution"] = "exponential"  # Use exponential distribution for arrivals
                # Run simulation with this configuration
                metrics = run_simulation(sim_config, reset_stats=True, traffic_pattern=traffic_pattern)
                
                # Store metrics for this iteration
                iteration_results.append({
                    'lam': lam,
                    'mu': mu,
                    'alpha': alpha,
                    'theta': theta,
                    'max_queue': max_queue,
                    'iteration': iteration,
                    'blocking_prob': metrics["blocking_prob"],
                    'waiting_time': metrics["waiting_time"],
                    'cpu_usage': metrics["cpu_usage"],
                    'ram_usage': metrics["ram_usage"],
                    'effective_arrival_rate': metrics["effective_arrival_rate"],
                    'waiting_requests': metrics["waiting_requests"],
                    'processing_requests': metrics["processing_requests"],
                    'traffic_pattern': traffic_pattern
                })
            
            # Add all iteration results to the main results list
            all_results.extend(iteration_results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save all raw results to CSV
    results_file = f"{results_dir}/optimization_results_raw_{timestamp}_{traffic_pattern}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Raw results saved to {results_file}")
    
    return results_df

def consolidate_and_plot_results(results_df=None, results_file=None, confidence_level=0.95, custom_weights=None, error_method='ci'):
    """
    Consolidates results across all configurations and creates a single plot 
    with error bars representing variation across configurations
    
    Parameters:
    -----------
    results_df : DataFrame, optional
        DataFrame containing all evaluation results. If None, results_file must be provided.
    results_file : str, optional
        Path to a saved results file to load instead of processing results_df.
    confidence_level : float
        Confidence level for interval calculation (default: 0.95 for 95% CI)
    custom_weights : dict, optional
        Custom weights for metrics when calculating optimal theta. 
        Example: {'blocking_prob': 0.4, 'waiting_time': 0.3, 'cpu_usage': 0.15, 'ram_usage': 0.15}
    error_method : str
        Method to represent error in plots: 'ci' for confidence intervals or 'percentile' for 10th-90th percentiles
    """
    # Create directory for plots if it doesn't exist
    plots_dir = "optimization_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for plot filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define default weights if not provided
    if custom_weights is None:
        custom_weights = {
            'blocking_prob': 0.25,    # Higher weight for blocking probability
            'waiting_time': 0.25,     # High weight for waiting time
            'cpu_usage': 0.25,       # Lower weight for CPU usage
            'ram_usage': 0.25        # Lower weight for RAM usage
        }
    
    # If results_file is provided, load from file instead of processing results_df
    if results_file:
        if os.path.exists(results_file):
            print(f"Loading consolidated results from {results_file}")
            cons_df = pd.read_csv(results_file)
        else:
            raise FileNotFoundError(f"Results file not found: {results_file}")
    else:
        # Process results_df to get consolidated results
        if results_df is None:
            raise ValueError("Either results_df or results_file must be provided")
        
        # Group results by theta only (combining all configurations)
        grouped = results_df.groupby(['theta'])
        
        # Calculate mean and error bars for each metric across all configurations
        consolidated_results = []
        
        for theta, group in grouped:
            # For each metric, calculate mean and error bounds based on selected method
            metrics = ['blocking_prob', 'waiting_time', 'cpu_usage', 'ram_usage']
            result = {'theta': theta}
            
            for metric in metrics:
                values = group[metric].values
                mean = np.mean(values)
                
                # Calculate error bounds based on selected method
                if error_method == 'ci':
                    # Calculate confidence interval
                    if len(values) > 1:  # Need at least 2 samples for t-distribution
                        t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(values)-1)
                        std_err = stats.sem(values)
                        margin = t_critical * std_err
                        lower_bound = mean - margin
                        upper_bound = mean + margin
                    else:
                        lower_bound = mean
                        upper_bound = mean
                else:  # percentile method
                    # Calculate 10th and 90th percentiles
                    if len(values) > 1:
                        lower_bound = np.percentile(values, 10)
                        upper_bound = np.percentile(values, 90)
                    else:
                        lower_bound = mean
                        upper_bound = mean
                
                result[f'{metric}_mean'] = mean
                result[f'{metric}_lower'] = lower_bound
                result[f'{metric}_upper'] = upper_bound
            
            consolidated_results.append(result)
        
        # Convert to DataFrame and sort by theta
        cons_df = pd.DataFrame(consolidated_results)
        cons_df = cons_df.sort_values('theta')
        
        # Save consolidated results
        cons_file = f"optimization_results/consolidated_results_{timestamp}.csv"
        cons_df.to_csv(cons_file, index=False)
        print(f"Consolidated results saved to {cons_file}")
    
    # Calculate weighted scores for each theta value
    theta_values = cons_df['theta'].values
    metrics = ['blocking_prob', 'waiting_time', 'cpu_usage', 'ram_usage']
    
    # First normalize each metric
    for metric in metrics:
        mean_values = cons_df[f'{metric}_mean'].values
        if np.max(mean_values) > np.min(mean_values):
            # Min-max normalization for each metric
            norm_values = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))
        else:
            norm_values = np.ones_like(mean_values) * 0.5
        
        # Add normalized values to dataframe
        cons_df[f'{metric}_norm'] = norm_values
    
    # Calculate weighted score (lower is better)
    weighted_scores = np.zeros_like(theta_values, dtype=float)
    for metric in metrics:
        weighted_scores += custom_weights[metric] * cons_df[f'{metric}_norm'].values
    
    # Add weighted scores to dataframe
    cons_df['weighted_score'] = weighted_scores
    
    # Find optimal theta value
    best_idx = np.argmin(weighted_scores)
    best_theta = theta_values[best_idx]
    best_score = weighted_scores[best_idx]
    
    # Define figure style variables
    fig_style = {
        "figure_size": (12, 8),
        "title_fontsize": 14,
        "label_fontsize": 24,
        "tick_fontsize": 24,
        "legend_fontsize": 18,
        "line_width": 3,
        "alpha": 0.15,  # Alpha for confidence interval regions
        "dpi": 300
    }
    
    # Create a single combined plot with all metrics
    plt.figure(figsize=(14, 10))
    
    # Plot each metric with its error bounds
    metrics_display = {
        'blocking_prob': ('Blocking Probability', 'blue'),
        'waiting_time': ('Waiting Time', 'red'),
        'cpu_usage': ('CPU Usage', 'green'),
        'ram_usage': ('RAM Usage', 'purple')
    }
    
    # Determine the error representation label for the plot title
    error_label = "95% Confidence Intervals" if error_method == 'ci' else "10th-90th Percentiles"
    
    # Plot each normalized metric with its error bounds
    for metric, (display_name, color) in metrics_display.items():
        if metric == 'cpu_usage':
            continue
        # Get data for this metric
        mean_values = cons_df[f'{metric}_mean'].values
        lower_values = cons_df[f'{metric}_lower'].values
        upper_values = cons_df[f'{metric}_upper'].values
        
        # Normalize the metric to [0,1] for better comparison
        if np.max(mean_values) > np.min(mean_values):
            norm_mean = (mean_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))
            norm_lower = (lower_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))
            norm_upper = (upper_values - np.min(mean_values)) / (np.max(mean_values) - np.min(mean_values))
        else:
            norm_mean = np.ones_like(mean_values) * 0.5
            norm_lower = np.ones_like(lower_values) * 0.5
            norm_upper = np.ones_like(upper_values) * 0.5
            
        # Plot mean line and error region
        plt.plot(theta_values, norm_mean, '-', color=color, linewidth=fig_style["line_width"], label=display_name)
        plt.fill_between(theta_values, norm_lower, norm_upper, color=color, alpha=fig_style["alpha"])
    
    # Plot weighted score line
    plt.plot(theta_values, weighted_scores, 'k--', linewidth=fig_style["line_width"]+1, label='Weighted Score')
    
    # Mark the optimal theta
    plt.plot(best_theta, best_score, 'ro', markersize=10)
    
    plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Normalized Metric Value', fontsize=fig_style["label_fontsize"])
    # plt.title(f'Consolidated Performance Metrics with {error_label}', fontsize=fig_style["label_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    plt.ylim(-0.1, 1.1)  # Set y-axis range from 0 to 1
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fig_style["legend_fontsize"], bbox_to_anchor=(0.05, 0.8), loc='center left')

    # Save consolidated plot
    plot_file = f"{plots_dir}/consolidated_metrics_{timestamp}.png"
    plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    
    print(f"Consolidated plot saved to {plot_file}")
    print(f"Optimal timeout rate (θ): {best_theta:.2f}")
    
    
    print(f"Individual metric plots saved to {plots_dir} directory.")
    
    return cons_df, best_theta

def evaluate_theta_impact(traffic_pattern='poisson'):
    """
    Evaluates the impact of different container idle timeout rates (theta)
    on system performance metrics.
    
    Parameters:
    -----------
    traffic_pattern : str
        Traffic pattern type: 'poisson' or 'day_night'
        
    Returns:
    --------
    metrics_dict : dict
        Dictionary containing theta values and corresponding metrics
    best_theta : float
        The optimal theta value identified
    best_metrics : dict
        Dictionary of metrics at the best theta value
    """
    print(f"Evaluating theta impact with {traffic_pattern} traffic pattern...")
    
    # Create directory for results
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for result filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define theta values to test
    theta_values = np.linspace(0.01, 1.0, 10)  # 10 values between 0.01 and 1.0
    
    # Create arrays to store metrics for each theta value
    blocking_probs = []
    waiting_times = []
    cpu_usages = []
    ram_usages = []
    
    # Create a baseline configuration for testing
    sim_config = copy.deepcopy(base_config)
    
    # Set fixed parameters
    sim_config["request"]["ram_warm"] = 10
    sim_config["request"]["cpu_warm"] = 1
    sim_config["request"]["ram_demand"] = 25
    sim_config["request"]["cpu_demand"] = 30
    sim_config["system"]["num_servers"] = 3
    sim_config["system"]["sim_time"] = 1000
    sim_config["system"]["verbose"] = False
    sim_config["request"]["arrival_rate"] = 3.0
    sim_config["request"]["service_rate"] = 1.0
    sim_config["container"]["spawn_time"] = 0.5
    sim_config["system"]["distribution"] = "exponential"
    
    # For each theta value
    for i, theta in enumerate(theta_values):
        print(f"  Processing theta = {theta:.2f} ({i+1}/{len(theta_values)})")
        
        # Set the idle_timeout based on the current theta
        # theta is timeout rate, so idle_timeout = 1/theta
        sim_config["container"]["idle_timeout"] = 1/theta
        
        # Run simulation with this configuration
        metrics = run_simulation(sim_config, reset_stats=True, traffic_pattern=traffic_pattern)
        
        # Store metrics for this theta value
        blocking_probs.append(metrics["blocking_prob"])
        waiting_times.append(metrics["waiting_time"])
        cpu_usages.append(metrics["cpu_usage"])
        ram_usages.append(metrics["ram_usage"])
    
    # Compile all metrics into a dictionary
    metrics_dict = {
        'theta_values': theta_values,
        'blocking_probs': blocking_probs,
        'waiting_times': waiting_times,
        'cpu_usages': cpu_usages,
        'ram_usages': ram_usages
    }
    
    # Save raw results to CSV
    results_df = pd.DataFrame({
        'theta': theta_values,
        'blocking_prob': blocking_probs,
        'waiting_time': waiting_times,
        'cpu_usage': cpu_usages,
        'ram_usage': ram_usages
    })
    
    results_file = f"{results_dir}/theta_impact_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Find best theta value based on metrics
    best_theta, best_metrics, ranking = find_best_theta(metrics_dict)
    
    # Plot the results
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_optimization_results(theta_values, ranking, best_theta, plots_dir, timestamp)
    
    return metrics_dict, best_theta, best_metrics

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate serverless simulator with different configurations')
    parser.add_argument('--mode', choices=['full', 'plot_only', 'theta'], default='theta',
                      help='Run mode: full evaluation, plot from existing results, or theta impact evaluation')
    parser.add_argument('--results-file', type=str, 
                      help='Path to saved results file (for plot_only mode)')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of iterations for confidence interval calculation')
    parser.add_argument('--error', choices=['ci', 'percentile'], default='ci',
                      help='Error representation method: confidence interval (ci) or 10th-90th percentile (percentile)')
    parser.add_argument('--traffic', choices=['poisson', 'day_night'], default='poisson',
                      help='Traffic pattern type: poisson (constant rate) or day_night (time-varying)')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        print("Starting multi-configuration evaluation with confidence intervals...")
        # Run evaluations across different configurations
        results_df = evaluate_multiple_configs(n_iterations=args.iterations, traffic_pattern=args.traffic)
        # Consolidate and plot results
        cons_df, best_theta = consolidate_and_plot_results(results_df=results_df, confidence_level=0.95, error_method=args.error)
        print(f"Multi-configuration evaluation complete! Optimal theta: {best_theta:.4f}")
    
    elif args.mode == 'plot_only':
        if not args.results_file:
            print("Error: --results-file must be specified in plot_only mode")
            parser.print_help()
            exit(1)
        
        print(f"Plotting results from {args.results_file}...")
        # Custom weights can be adjusted here if needed
        custom_weights = {'blocking_prob': 0.4, 'waiting_time': 0.3, 'cpu_usage': 0.15, 'ram_usage': 0.15}
        cons_df, best_theta = consolidate_and_plot_results(
            results_file=args.results_file, 
            confidence_level=0.95,
            custom_weights=custom_weights,
            error_method=args.error
        )
        print(f"Plotting complete! Optimal theta: {best_theta:.4f}")
    
    elif args.mode == 'theta':
        print("Running theta impact evaluation...")
        metrics_dict, best_theta, best_metrics = evaluate_theta_impact(traffic_pattern=args.traffic)
        print(f"Theta evaluation complete! Optimal theta: {best_theta:.4f}")