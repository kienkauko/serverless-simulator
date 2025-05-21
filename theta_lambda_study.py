#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the serverless simulator to study the relationship
between arrival rate (lambda) and optimal timeout rate (theta)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pandas as pd
import simpy
import copy
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
    
    # Compile metrics
    metrics = {
        "blocking_prob": blocking_prob,
        "waiting_time": avg_waiting_time,
        "cpu_usage": mean_cpu_usage,
        "ram_usage": mean_ram_usage
    }
    
    return metrics

def run_simulation_with_repetitions(sim_config, theta_values, num_repetitions=5, traffic_pattern='poisson'):
    """
    Run multiple simulations with the provided configuration and return the average metrics
    
    Parameters:
    -----------
    sim_config : dict
        Dictionary containing simulator configuration
    theta_values : array-like
        Array of theta values to test
    num_repetitions : int
        Number of repetitions to run for statistical significance
    traffic_pattern : str
        Traffic pattern type: 'poisson' or 'day_night'
        
    Returns:
    --------
    mean_metrics : dict
        Dictionary containing the mean simulation metrics for each theta
    std_metrics : dict
        Dictionary containing the standard deviation of metrics for each theta
    best_theta_stats : dict
        Dictionary containing statistics about the best theta values from repetitions
    """
    # Arrays to store metrics for each theta value
    mean_blocking_probs = []
    mean_waiting_times = []
    mean_cpu_usages = []
    mean_ram_usages = []
    
    std_blocking_probs = []
    std_waiting_times = []
    std_cpu_usages = []
    std_ram_usages = []
    
    # Arrays to track best theta values from each repetition
    best_thetas_overall = []  # Track best theta for each repetition
    best_thetas_blocking = []
    best_thetas_waiting = []
    best_thetas_cpu = []
    best_thetas_ram = []
    
    # For each theta value
    for theta_idx, theta in enumerate(theta_values):
        print(f"  Testing theta = {theta:.2f} ({theta_idx+1}/{len(theta_values)})")
        
        # Set the idle_timeout based on the current theta
        # theta is timeout rate, so idle_timeout = 1/theta
        sim_config["container"]["idle_timeout"] = 1/theta
        
        # Initialize lists to store metrics from each repetition for this theta
        blocking_probs = []
        waiting_times = []
        cpu_usages = []
        ram_usages = []
        
        # Run the simulation multiple times for this theta
        for i in range(num_repetitions):
            # Run a single simulation
            metrics = run_simulation(sim_config, reset_stats=True, traffic_pattern=traffic_pattern)
            
            # Store metrics from this repetition
            blocking_probs.append(metrics["blocking_prob"])
            waiting_times.append(metrics["waiting_time"])
            cpu_usages.append(metrics["cpu_usage"])
            ram_usages.append(metrics["ram_usage"])
        
        # Calculate mean metrics for this theta
        mean_blocking_probs.append(np.mean(blocking_probs))
        mean_waiting_times.append(np.mean(waiting_times))
        mean_cpu_usages.append(np.mean(cpu_usages))
        mean_ram_usages.append(np.mean(ram_usages))
        
        # Calculate standard deviation of metrics for this theta
        std_blocking_probs.append(np.std(blocking_probs))
        std_waiting_times.append(np.std(waiting_times))
        std_cpu_usages.append(np.std(cpu_usages))
        std_ram_usages.append(np.std(ram_usages))
    
    # Compile metrics dictionary
    mean_metrics = {
        'blocking_probs': mean_blocking_probs,
        'waiting_times': mean_waiting_times,
        'cpu_usages': mean_cpu_usages,
        'ram_usages': mean_ram_usages
    }
    
    std_metrics = {
        'blocking_probs': std_blocking_probs,
        'waiting_times': std_waiting_times,
        'cpu_usages': std_cpu_usages,
        'ram_usages': std_ram_usages
    }
    
    # For each repetition, determine the best theta for each metric
    for rep in range(num_repetitions):
        # Create arrays for metrics from this repetition for all theta values
        rep_blocking_probs = []
        rep_waiting_times = []
        rep_cpu_usages = []
        rep_ram_usages = []
        
        # For each theta, get the metrics from this repetition
        for theta_idx, theta in enumerate(theta_values):
            # Set the idle_timeout based on the current theta
            sim_config["container"]["idle_timeout"] = 1/theta
            
            # Run a single simulation for this repetition and theta
            metrics = run_simulation(sim_config, reset_stats=True, traffic_pattern=traffic_pattern)
            
            # Store the metrics
            rep_blocking_probs.append(metrics["blocking_prob"])
            rep_waiting_times.append(metrics["waiting_time"])
            rep_cpu_usages.append(metrics["cpu_usage"])
            rep_ram_usages.append(metrics["ram_usage"])
        
        # Create metrics dictionary for this repetition
        rep_metrics_dict = {
            'theta_values': theta_values,
            'blocking_probs': rep_blocking_probs,
            'waiting_times': rep_waiting_times,
            'cpu_usages': rep_cpu_usages,
            'ram_usages': rep_ram_usages
        }
        
        # Find best theta for this repetition
        best_theta_overall, _, _ = find_best_theta(rep_metrics_dict)
        best_thetas_overall.append(best_theta_overall)
        
        # Find best theta for individual metrics in this repetition
        best_idx_blocking = np.argmin(rep_blocking_probs)
        best_thetas_blocking.append(theta_values[best_idx_blocking])
        
        best_idx_waiting = np.argmin(rep_waiting_times)
        best_thetas_waiting.append(theta_values[best_idx_waiting])
        
        best_idx_cpu = np.argmin(rep_cpu_usages)
        best_thetas_cpu.append(theta_values[best_idx_cpu])
        
        best_idx_ram = np.argmin(rep_ram_usages)
        best_thetas_ram.append(theta_values[best_idx_ram])
    
    # Calculate statistics for best theta values
    best_theta_stats = {
        'overall': {
            'mean': np.mean(best_thetas_overall),
            'std': np.std(best_thetas_overall)
        },
        'blocking': {
            'mean': np.mean(best_thetas_blocking),
            'std': np.std(best_thetas_blocking)
        },
        'waiting': {
            'mean': np.mean(best_thetas_waiting),
            'std': np.std(best_thetas_waiting)
        },
        'cpu': {
            'mean': np.mean(best_thetas_cpu),
            'std': np.std(best_thetas_cpu)
        },
        'ram': {
            'mean': np.mean(best_thetas_ram),
            'std': np.std(best_thetas_ram)
        }
    }
    
    return mean_metrics, std_metrics, best_theta_stats

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
            'blocking_probs': 0.25,    # Higher weight for blocking probability
            'waiting_times': 0.25,     # High weight for waiting time
            'cpu_usages': 0.25,       # Lower weight for CPU usage
            'ram_usages': 0.25        # Lower weight for RAM usage
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

def evaluate_lambda_theta_relationship(traffic_pattern='poisson', num_repetitions=5):
    """
    Evaluates how the optimal container idle timeout rate (theta) varies
    with different request arrival rates (lambda) for each individual metric.
    
    Parameters:
    -----------
    traffic_pattern : str
        Traffic pattern type: 'poisson' or 'day_night'
    num_repetitions : int
        Number of repetitions to run for each configuration for statistical significance
        
    Returns:
    --------
    lambda_theta_df : DataFrame
        DataFrame containing arrival rates and their corresponding optimal theta values for each metric
    """
    print(f"Evaluating lambda-theta relationship with {traffic_pattern} traffic pattern...")
    
    # Create directory for results
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for result filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define arrival rate (lambda) values to test
    lambda_values = np.linspace(1.0, 10.0, 10)  # 10 values between 1.0 and 10.0
    
    # Define theta values to test for each lambda
    theta_values = np.linspace(0.01, 1.0, 10)  # 10 values between 0.01 and 1.0
    
    # Create arrays to store best theta for each lambda value and each metric
    best_thetas_overall = []  # Best theta overall (weighted combination)
    best_thetas_blocking = []  # Best theta for minimizing blocking probability
    best_thetas_waiting = []   # Best theta for minimizing waiting time
    best_thetas_cpu = []       # Best theta for minimizing CPU usage
    best_thetas_ram = []       # Best theta for minimizing RAM usage
    
    # Arrays to store metric values at best theta for each lambda
    best_blocking_probs = []
    best_waiting_times = []
    best_cpu_usages = []
    best_ram_usages = []
    
    # Arrays to store errors (standard deviations) for metrics and best thetas
    best_blocking_errors = []
    best_waiting_errors = []
    best_cpu_errors = []
    best_ram_errors = []
    best_overall_errors = []  # Error for the overall best theta
    
    # Create a baseline configuration for testing
    sim_config = copy.deepcopy(base_config)
    
    # Set fixed parameters
    sim_config["request"]["ram_warm"] = 10
    sim_config["request"]["cpu_warm"] = 1
    sim_config["request"]["ram_demand"] = 25
    sim_config["request"]["cpu_demand"] = 30
    sim_config["system"]["num_servers"] = 10
    sim_config["system"]["sim_time"] = 500
    sim_config["system"]["verbose"] = False
    sim_config["request"]["service_rate"] = 1.0
    sim_config["container"]["spawn_time"] = 0.5
    sim_config["system"]["distribution"] = "exponential"  # Exponential distribution for request arrivals
    
    # For each lambda value
    for lambda_idx, lam in enumerate(lambda_values):
        print(f"Processing lambda = {lam:.2f} ({lambda_idx+1}/{len(lambda_values)})")
        
        # Set the arrival rate
        sim_config["request"]["arrival_rate"] = lam
        
        # Run simulation with multiple repetitions for this configuration
        mean_metrics, std_metrics, best_theta_stats = run_simulation_with_repetitions(
            sim_config, theta_values, num_repetitions=num_repetitions, traffic_pattern=traffic_pattern)
        
        # Store the mean best theta values and their standard deviations
        best_thetas_overall.append(best_theta_stats['overall']['mean'])
        best_overall_errors.append(best_theta_stats['overall']['std'])
        
        best_thetas_blocking.append(best_theta_stats['blocking']['mean'])
        best_theta_blocking_error = best_theta_stats['blocking']['std']
        
        best_thetas_waiting.append(best_theta_stats['waiting']['mean'])
        best_theta_waiting_error = best_theta_stats['waiting']['std']
        
        best_thetas_cpu.append(best_theta_stats['cpu']['mean'])
        best_theta_cpu_error = best_theta_stats['cpu']['std']
        
        best_thetas_ram.append(best_theta_stats['ram']['mean'])
        best_theta_ram_error = best_theta_stats['ram']['std']
        
        # Compile metrics dictionary for this lambda value
        metrics_dict = {
            'theta_values': theta_values,
            'blocking_probs': mean_metrics['blocking_probs'],
            'waiting_times': mean_metrics['waiting_times'],
            'cpu_usages': mean_metrics['cpu_usages'],
            'ram_usages': mean_metrics['ram_usages']
        }
        
        # Find the best metrics at the average best theta values
        # For blocking probability (lower is better)
        best_idx_blocking = np.argmin(mean_metrics['blocking_probs'])
        best_blocking_probs.append(mean_metrics['blocking_probs'][best_idx_blocking])
        best_blocking_errors.append(std_metrics['blocking_probs'][best_idx_blocking])
        
        # For waiting time (lower is better)
        best_idx_waiting = np.argmin(mean_metrics['waiting_times'])
        best_waiting_times.append(mean_metrics['waiting_times'][best_idx_waiting])
        best_waiting_errors.append(std_metrics['waiting_times'][best_idx_waiting])
        
        # For CPU usage (lower is better)
        best_idx_cpu = np.argmin(mean_metrics['cpu_usages'])
        best_cpu_usages.append(mean_metrics['cpu_usages'][best_idx_cpu])
        best_cpu_errors.append(std_metrics['cpu_usages'][best_idx_cpu])
        
        # For RAM usage (lower is better)
        best_idx_ram = np.argmin(mean_metrics['ram_usages'])
        best_ram_usages.append(mean_metrics['ram_usages'][best_idx_ram])
        best_ram_errors.append(std_metrics['ram_usages'][best_idx_ram])
        
        print(f"  Best theta (overall) for lambda={lam:.2f}: {best_theta_stats['overall']['mean']:.4f} ± {best_theta_stats['overall']['std']:.4f}")
        print(f"  Best theta (blocking) for lambda={lam:.2f}: {best_theta_stats['blocking']['mean']:.4f} ± {best_theta_stats['blocking']['std']:.4f}")
        print(f"  Best theta (waiting) for lambda={lam:.2f}: {best_theta_stats['waiting']['mean']:.4f} ± {best_theta_stats['waiting']['std']:.4f}")
        print(f"  Best theta (CPU) for lambda={lam:.2f}: {best_theta_stats['cpu']['mean']:.4f} ± {best_theta_stats['cpu']['std']:.4f}")
        print(f"  Best theta (RAM) for lambda={lam:.2f}: {best_theta_stats['ram']['mean']:.4f} ± {best_theta_stats['ram']['std']:.4f}")
    
    # Compile results into a DataFrame
    lambda_theta_df = pd.DataFrame({
        'lambda': lambda_values,
        'best_theta_overall': best_thetas_overall,
        'best_theta_blocking': best_thetas_blocking,
        'best_theta_waiting': best_thetas_waiting,
        'best_theta_cpu': best_thetas_cpu,
        'best_theta_ram': best_thetas_ram,
        'best_blocking_prob': best_blocking_probs,
        'best_waiting_time': best_waiting_times,
        'best_cpu_usage': best_cpu_usages,
        'best_ram_usage': best_ram_usages,
        'best_blocking_error': best_blocking_errors,
        'best_waiting_error': best_waiting_errors,
        'best_cpu_error': best_cpu_errors,
        'best_ram_error': best_ram_errors,
        'best_overall_error': best_overall_errors  # Error for the overall best theta
    })
    
    # Save results to CSV
    results_file = f"{results_dir}/lambda_theta_relationship_{timestamp}.csv"
    lambda_theta_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Plot lambda-theta relationship for each metric
    plot_lambda_theta_relationship_by_metric(lambda_theta_df, timestamp)
    
    # Plot metrics at best theta with confidence intervals
    # plot_metrics_at_best_theta_with_ci(lambda_theta_df, timestamp)
    
    return lambda_theta_df

def plot_lambda_theta_relationship(lambda_theta_df, timestamp=None):
    """
    Plot the relationship between arrival rate (lambda) and optimal timeout rate (theta)
    
    Parameters:
    -----------
    lambda_theta_df : DataFrame
        DataFrame containing lambda values and corresponding best theta values
    timestamp : str, optional
        Timestamp to use in filenames
    """
    # Create directory for plots if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define figure style variables for consistent visualization
    fig_style = {
        "figure_size": (10, 8),            # Figure size in inches
        "title_fontsize": 18,              # Title font size
        "label_fontsize": 16,              # Axis label font size
        "tick_fontsize": 14,               # Tick label font size
        "legend_fontsize": 14,             # Legend font size
        "line_width": 3,                   # Line width for plots
        "marker_size": 8,                  # Marker size
        "grid_alpha": 0.3,                 # Grid transparency
        "dpi": 300                         # DPI for saved figures
    }
    
    # Create figure
    plt.figure(figsize=fig_style["figure_size"])
    
    # Extract data
    lambda_values = lambda_theta_df['lambda'].values
    best_thetas = lambda_theta_df['best_theta'].values
    
    # Plot lambda-theta relationship
    plt.plot(best_thetas, lambda_values, 'b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"])
    
    # Add polynomial fit to visualize trend
    z = np.polyfit(best_thetas, lambda_values, 3)
    p = np.poly1d(z)
    theta_range = np.linspace(min(best_thetas), max(best_thetas), 100)
    plt.plot(theta_range, p(theta_range), 'r--', linewidth=fig_style["line_width"]-1,
             label='Polynomial Fit')
    
    # Label axes
    plt.xlabel('Optimal Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.title('Relationship Between Arrival Rate and Optimal Timeout Rate', 
              fontsize=fig_style["title_fontsize"])
    
    # Set tick font sizes
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    
    # Add grid and legend
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    
    # Save the plot
    plot_file = f"{plots_dir}/lambda_theta_relationship_{timestamp}.png"
    # plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    
    print(f"Lambda-theta relationship plot saved to {plot_file}")

def plot_metrics_at_best_theta(lambda_theta_df, timestamp=None):
    """
    Plot individual metrics at the best theta for each lambda
    
    Parameters:
    -----------
    lambda_theta_df : DataFrame
        DataFrame containing lambda values and corresponding best theta values
    timestamp : str, optional
        Timestamp to use in filenames
    """
    # Create directory for plots if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp is datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define figure style variables for consistent visualization
    fig_style = {
        "figure_size": (10, 8),            # Figure size in inches
        "title_fontsize": 18,              # Title font size
        "label_fontsize": 16,              # Axis label font size
        "tick_fontsize": 14,               # Tick label font size
        "legend_fontsize": 14,             # Legend font size
        "line_width": 3,                   # Line width for plots
        "marker_size": 8,                  # Marker size
        "grid_alpha": 0.3,                 # Grid transparency
        "dpi": 300                         # DPI for saved figures
    }
    
    # Extract data
    lambda_values = lambda_theta_df['lambda'].values
    blocking_probs = lambda_theta_df['blocking_prob'].values
    waiting_times = lambda_theta_df['waiting_time'].values
    cpu_usages = lambda_theta_df['cpu_usage'].values
    ram_usages = lambda_theta_df['ram_usage'].values
    
    # Plot blocking probability
    plt.figure(figsize=fig_style["figure_size"])
    plt.plot(lambda_values, blocking_probs, 'b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"])
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Blocking Probability', fontsize=fig_style["label_fontsize"])
    plt.title('Blocking Probability at Best Theta', fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plot_file = f"{plots_dir}/blocking_prob_{timestamp}.png"
    # plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    print(f"Blocking probability plot saved to {plot_file}")
    
    # Plot waiting time
    plt.figure(figsize=fig_style["figure_size"])
    plt.plot(lambda_values, waiting_times, 'b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"])
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Waiting Time', fontsize=fig_style["label_fontsize"])
    plt.title('Waiting Time at Best Theta', fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plot_file = f"{plots_dir}/waiting_time_{timestamp}.png"
    # plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    print(f"Waiting time plot saved to {plot_file}")
    
    # Plot CPU usage
    plt.figure(figsize=fig_style["figure_size"])
    plt.plot(lambda_values, cpu_usages, 'b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"])
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('CPU Usage', fontsize=fig_style["label_fontsize"])
    plt.title('CPU Usage at Best Theta', fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plot_file = f"{plots_dir}/cpu_usage_{timestamp}.png"
    # plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    print(f"CPU usage plot saved to {plot_file}")
    
    # Plot RAM usage
    plt.figure(figsize=fig_style["figure_size"])
    plt.plot(lambda_values, ram_usages, 'b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"])
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('RAM Usage', fontsize=fig_style["label_fontsize"])
    plt.title('RAM Usage at Best Theta', fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plot_file = f"{plots_dir}/ram_usage_{timestamp}.png"
    # plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    print(f"RAM usage plot saved to {plot_file}")

def plot_lambda_theta_relationship_by_metric(lambda_theta_df, timestamp=None):
    """
    Plot the relationship between arrival rate (lambda) and optimal timeout rate (theta)
    for each individual metric with confidence intervals
    
    Parameters:
    -----------
    lambda_theta_df : DataFrame
        DataFrame containing lambda values and corresponding best theta values for each metric
    timestamp : str, optional
        Timestamp to use in filenames
    """
    # Create directory for plots if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define figure style variables for consistent visualization
    fig_style = {
        "figure_size": (12, 10),           # Figure size in inches
        "title_fontsize": 18,              # Title font size
        "label_fontsize": 16,              # Axis label font size
        "tick_fontsize": 14,               # Tick label font size
        "legend_fontsize": 14,             # Legend font size
        "line_width": 3,                   # Line width for plots
        "marker_size": 8,                  # Marker size
        "grid_alpha": 0.3,                 # Grid transparency
        "dpi": 300                         # DPI for saved figures
    }
    
    # Extract data
    lambda_values = lambda_theta_df['lambda'].values
    best_thetas_overall = lambda_theta_df['best_theta_overall'].values
    best_thetas_blocking = lambda_theta_df['best_theta_blocking'].values
    best_thetas_waiting = lambda_theta_df['best_theta_waiting'].values
    best_thetas_cpu = lambda_theta_df['best_theta_cpu'].values
    best_thetas_ram = lambda_theta_df['best_theta_ram'].values
    
    # Calculate 95% confidence interval multiplier (using t-distribution with 4 degrees of freedom for 5 samples)
    # For 95% CI with 4 degrees of freedom (5 samples), t-value is approximately 2.776
    t_value = 2.776
    
    # Create figure for combined plot of all metrics with confidence intervals
    plt.figure(figsize=fig_style["figure_size"])
    
    # Plot best theta for each metric with confidence bands
    # For overall theta
    plt.plot(lambda_values, best_thetas_overall, 'k-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], label='Overall Best θ')
             
    # Add confidence interval for overall theta
    if 'best_overall_error' in lambda_theta_df.columns:
        overall_errors = lambda_theta_df['best_overall_error'].values * t_value
        plt.fill_between(lambda_values, 
                         best_thetas_overall - overall_errors, 
                         best_thetas_overall + overall_errors, 
                         color='k', alpha=0.1)
    
    # For blocking probability theta
    plt.plot(lambda_values, best_thetas_blocking, 'b-s', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], label='Best θ for Blocking Prob')
             
    # For waiting time theta
    plt.plot(lambda_values, best_thetas_waiting, 'r-^', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], label='Best θ for Waiting Time')
             
    # For CPU usage theta
    plt.plot(lambda_values, best_thetas_cpu, 'g-d', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], label='Best θ for CPU Usage')
             
    # For RAM usage theta
    plt.plot(lambda_values, best_thetas_ram, 'm-*', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], label='Best θ for RAM Usage')
             
    # Label axes
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Optimal Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    plt.title('Optimal Timeout Rate (θ) for Different Metrics by Arrival Rate (λ) with 95% CI', 
              fontsize=fig_style["title_fontsize"])
    
    # Set tick font sizes
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    
    # Add grid and legend
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    
    # Show the plot (but don't save it as requested)
    plt.show()
    
    print("Lambda-theta relationship plot by metric with confidence intervals displayed")

def plot_metrics_at_best_theta_with_ci(lambda_theta_df, timestamp=None):
    """
    Plot individual metrics at the best theta for each lambda with 95% confidence intervals
    
    Parameters:
    -----------
    lambda_theta_df : DataFrame
        DataFrame containing lambda values and corresponding metrics with errors
    timestamp : str, optional
        Timestamp to use in filenames
    """
    # Create directory for plots if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp is datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define figure style variables for consistent visualization
    fig_style = {
        "figure_size": (12, 10),           # Figure size in inches
        "title_fontsize": 18,              # Title font size
        "label_fontsize": 16,              # Axis label font size
        "tick_fontsize": 14,               # Tick label font size
        "legend_fontsize": 14,             # Legend font size
        "line_width": 3,                   # Line width for plots
        "marker_size": 8,                  # Marker size
        "grid_alpha": 0.3,                 # Grid transparency
        "dpi": 300                         # DPI for saved figures
    }
    
    # Extract data
    lambda_values = lambda_theta_df['lambda'].values
    
    # Calculate 95% confidence interval multiplier (using t-distribution with 4 degrees of freedom for 5 samples)
    # For 95% CI with 4 degrees of freedom (5 samples), t-value is approximately 2.776
    t_value = 2.776
    
    # Create subplot for all metrics with confidence intervals
    fig, axs = plt.subplots(2, 2, figsize=(fig_style["figure_size"][0], fig_style["figure_size"][1]))
    fig.suptitle('Performance Metrics with 95% Confidence Intervals', fontsize=fig_style["title_fontsize"])
    
    # Plot blocking probability with confidence interval
    blocking_probs = lambda_theta_df['best_blocking_prob'].values
    blocking_errors = lambda_theta_df['best_blocking_error'].values * t_value
    axs[0, 0].errorbar(lambda_values, blocking_probs, yerr=blocking_errors, 
             fmt='b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='Blocking Probability')
    axs[0, 0].set_xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    axs[0, 0].set_ylabel('Blocking Probability', fontsize=fig_style["label_fontsize"])
    axs[0, 0].set_title('Blocking Probability at Best Theta', fontsize=fig_style["label_fontsize"])
    axs[0, 0].tick_params(labelsize=fig_style["tick_fontsize"])
    axs[0, 0].grid(True, alpha=fig_style["grid_alpha"])
    
    # Plot waiting time with confidence interval
    waiting_times = lambda_theta_df['best_waiting_time'].values
    waiting_errors = lambda_theta_df['best_waiting_error'].values * t_value
    axs[0, 1].errorbar(lambda_values, waiting_times, yerr=waiting_errors, 
             fmt='r-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='Waiting Time')
    axs[0, 1].set_xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    axs[0, 1].set_ylabel('Waiting Time', fontsize=fig_style["label_fontsize"])
    axs[0, 1].set_title('Waiting Time at Best Theta', fontsize=fig_style["label_fontsize"])
    axs[0, 1].tick_params(labelsize=fig_style["tick_fontsize"])
    axs[0, 1].grid(True, alpha=fig_style["grid_alpha"])
    
    # Plot CPU usage with confidence interval
    cpu_usages = lambda_theta_df['best_cpu_usage'].values
    cpu_errors = lambda_theta_df['best_cpu_error'].values * t_value
    axs[1, 0].errorbar(lambda_values, cpu_usages, yerr=cpu_errors, 
             fmt='g-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='CPU Usage')
    axs[1, 0].set_xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    axs[1, 0].set_ylabel('CPU Usage', fontsize=fig_style["label_fontsize"])
    axs[1, 0].set_title('CPU Usage at Best Theta', fontsize=fig_style["label_fontsize"])
    axs[1, 0].tick_params(labelsize=fig_style["tick_fontsize"])
    axs[1, 0].grid(True, alpha=fig_style["grid_alpha"])
    
    # Plot RAM usage with confidence interval
    ram_usages = lambda_theta_df['best_ram_usage'].values
    ram_errors = lambda_theta_df['best_ram_error'].values * t_value
    axs[1, 1].errorbar(lambda_values, ram_usages, yerr=ram_errors, 
             fmt='m-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='RAM Usage')
    axs[1, 1].set_xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    axs[1, 1].set_ylabel('RAM Usage', fontsize=fig_style["label_fontsize"])
    axs[1, 1].set_title('RAM Usage at Best Theta', fontsize=fig_style["label_fontsize"])
    axs[1, 1].tick_params(labelsize=fig_style["tick_fontsize"])
    axs[1, 1].grid(True, alpha=fig_style["grid_alpha"])
    
    # Adjust layout and show
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to accommodate suptitle
    plt.show()
    
    # Also create individual plots for each metric with confidence intervals
    
    # Blocking probability with CI
    plt.figure(figsize=fig_style["figure_size"])
    plt.errorbar(lambda_values, blocking_probs, yerr=blocking_errors, 
             fmt='b-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='Blocking Probability')
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Blocking Probability', fontsize=fig_style["label_fontsize"])
    plt.title('Blocking Probability at Best Theta with 95% Confidence Interval', 
              fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    plt.show()
    
    # Waiting time with CI
    plt.figure(figsize=fig_style["figure_size"])
    plt.errorbar(lambda_values, waiting_times, yerr=waiting_errors, 
             fmt='r-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='Waiting Time')
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Waiting Time', fontsize=fig_style["label_fontsize"])
    plt.title('Waiting Time at Best Theta with 95% Confidence Interval', 
              fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    plt.show()
    
    # CPU usage with CI
    plt.figure(figsize=fig_style["figure_size"])
    plt.errorbar(lambda_values, cpu_usages, yerr=cpu_errors, 
             fmt='g-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='CPU Usage')
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('CPU Usage', fontsize=fig_style["label_fontsize"])
    plt.title('CPU Usage at Best Theta with 95% Confidence Interval', 
              fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    plt.show()
    
    # RAM usage with CI
    plt.figure(figsize=fig_style["figure_size"])
    plt.errorbar(lambda_values, ram_usages, yerr=ram_errors, 
             fmt='m-o', linewidth=fig_style["line_width"], 
             markersize=fig_style["marker_size"], capsize=5, label='RAM Usage')
    plt.xlabel('Arrival Rate (λ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('RAM Usage', fontsize=fig_style["label_fontsize"])
    plt.title('RAM Usage at Best Theta with 95% Confidence Interval', 
              fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    plt.show()
    
    print("Metric plots with confidence intervals displayed")

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate the relationship between arrival rate and optimal timeout rate')
    parser.add_argument('--traffic', choices=['poisson', 'day_night'], default='poisson',
                      help='Traffic pattern type: poisson (constant rate) or day_night (time-varying)')
    
    args = parser.parse_args()
    
    print("Running lambda-theta relationship study...")
    lambda_theta_df = evaluate_lambda_theta_relationship(traffic_pattern=args.traffic)
    print(f"Lambda-theta relationship study complete!")