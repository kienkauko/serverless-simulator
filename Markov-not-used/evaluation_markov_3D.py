#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the Markov model with varying timeout rate (theta)
This script analyzes how different values of theta affect various metrics
across multiple system configurations
"""

import matplotlib.pyplot as plt
import numpy as np
from Markov.model_3D import MarkovModel
import os
import datetime
from scipy import stats
import pandas as pd
import itertools

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
    # plt.title('Normalized Metrics vs. Timeout Rate (θ)', fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.legend(fontsize=fig_style["legend_fontsize"])
    plt.grid(True, alpha=fig_style["grid_alpha"])
    # plt.savefig(f"{plots_dir}/optimization_results_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()

def evaluate_theta_impact():
    """
    Evaluates the impact of varying theta on different metrics
    Generates plots showing the relationship between theta and:
    - Blocking probability
    - Waiting time
    - CPU usage
    - RAM usage
    
    Returns:
    --------
    metrics_dict : dict
        Dictionary containing metrics data
    best_theta : float
        The optimal theta value
    best_metrics : dict
        Dictionary with metrics at the optimal theta value
    """
    # Define figure style variables for consistent visualization
    fig_style = {
        "figure_size": (10, 6),            # Figure size in inches
        "title_fontsize": 14,              # Title font size
        "label_fontsize": 12,              # Axis label font size
        "tick_fontsize": 10,               # Tick label font size
        "legend_fontsize": 10,             # Legend font size
        "line_width": 2,                   # Line width for plots
        "marker_size": 6,                  # Marker size
        "grid_alpha": 0.3,                 # Grid transparency
        "marker_style": 'o',               # Marker style
        "dpi": 300                         # DPI for saved figures
    }
    
    # Define base configuration with fixed values
    base_config = {
        "lam": 5,       # Arrival rate
        "mu": 5,         # Service rate
        "alpha": 0.1,      # Request activation rate
        "beta": 5,      # Idle encounter rate
        "theta": 0.0,      # Container timeout      rate (to be varied)
        "max_queue": 5,   # Maximum queue size
        "ram_warm": 10,     # RAM usage in warm state
        "cpu_warm": 1,     # CPU usage in warm state
        "ram_demand": 56,  # RAM usage in active state
        "cpu_demand": 21   # CPU usage in active state
    }
    
    # Define range of theta values to evaluate
    theta_values = np.linspace(0.0, 5.0, 10)  # From 0 to 2.0 with 20 points
    
    # Initialize storage for metrics
    blocking_probs = []
    waiting_times = []
    cpu_usages = []
    ram_usages = []
    effective_arrival_rates = []
    waiting_requests = []
    processing_requests = []
    
    print("Evaluating model with different theta values...")
    
    # Evaluate model for each theta value
    for theta in theta_values:
        print(f"Processing theta = {theta:.2f}")
        # Update config with current theta value
        config = base_config.copy()
        config["theta"] = theta
        
        # Create and evaluate model
        model = MarkovModel(config, verbose=False)
        metrics = model.get_metrics()
        
        # Store metrics
        blocking_probs.append(metrics["blocking_ratios"][0])
        waiting_times.append(metrics["waiting_times"][0])
        cpu_usages.append(metrics["mean_cpu_usage_per_request"][0])
        ram_usages.append(metrics["mean_ram_usage_per_request"][0])
        effective_arrival_rates.append(metrics["effective_arrival_rates"][0])
        waiting_requests.append(metrics["waiting_requests"][0])
        processing_requests.append(metrics["processing_requests"][0])
    
    # Create plots directory if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for plot filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot blocking probability vs theta
    # plt.figure(figsize=fig_style["figure_size"])
    # plt.plot(theta_values, blocking_probs, fig_style["marker_style"], linewidth=fig_style["line_width"], 
    #          markersize=fig_style["marker_size"], color='blue')
    # plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    # plt.ylabel('Blocking Probability', fontsize=fig_style["label_fontsize"])
    # plt.title('Impact of Timeout Rate on Blocking Probability', fontsize=fig_style["title_fontsize"])
    # plt.xticks(fontsize=fig_style["tick_fontsize"])
    # plt.yticks(fontsize=fig_style["tick_fontsize"])
    # plt.grid(True, alpha=fig_style["grid_alpha"])
    # plt.savefig(f"{plots_dir}/blocking_prob_vs_theta_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    # plt.show()
    
    # Plot waiting time vs theta
    # plt.figure(figsize=fig_style["figure_size"])
    # plt.plot(theta_values, waiting_times, fig_style["marker_style"], linewidth=fig_style["line_width"], 
    #          markersize=fig_style["marker_size"], color='red')
    # plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    # plt.ylabel('Waiting Time', fontsize=fig_style["label_fontsize"])
    # plt.title('Impact of Timeout Rate on Waiting Time', fontsize=fig_style["title_fontsize"])
    # plt.xticks(fontsize=fig_style["tick_fontsize"])
    # plt.yticks(fontsize=fig_style["tick_fontsize"])
    # plt.grid(True, alpha=fig_style["grid_alpha"])
    # plt.savefig(f"{plots_dir}/waiting_time_vs_theta_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    # plt.show()
    
    # Plot CPU usage vs theta
    # plt.figure(figsize=fig_style["figure_size"])
    # plt.plot(theta_values, cpu_usages, fig_style["marker_style"], linewidth=fig_style["line_width"], 
    #          markersize=fig_style["marker_size"], color='green')
    # plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    # plt.ylabel('CPU Usage', fontsize=fig_style["label_fontsize"])
    # plt.title('Impact of Timeout Rate on CPU Usage', fontsize=fig_style["title_fontsize"])
    # plt.xticks(fontsize=fig_style["tick_fontsize"])
    # plt.yticks(fontsize=fig_style["tick_fontsize"])
    # plt.grid(True, alpha=fig_style["grid_alpha"])
    # plt.savefig(f"{plots_dir}/cpu_usage_vs_theta_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    # plt.show()
    
    # Plot RAM usage vs theta
    # plt.figure(figsize=fig_style["figure_size"])
    # plt.plot(theta_values, ram_usages, fig_style["marker_style"], linewidth=fig_style["line_width"], 
    #          markersize=fig_style["marker_size"], color='magenta')
    # plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    # plt.ylabel('RAM Usage', fontsize=fig_style["label_fontsize"])
    # plt.title('Impact of Timeout Rate on RAM Usage', fontsize=fig_style["title_fontsize"])
    # plt.xticks(fontsize=fig_style["tick_fontsize"])
    # plt.yticks(fontsize=fig_style["tick_fontsize"])
    # plt.grid(True, alpha=fig_style["grid_alpha"])
    # plt.savefig(f"{plots_dir}/ram_usage_vs_theta_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    # plt.show()
    
    # Save results to CSV
    results = np.column_stack((
        theta_values, 
        blocking_probs, 
        waiting_times, 
        cpu_usages, 
        ram_usages, 
        effective_arrival_rates, 
        waiting_requests, 
        processing_requests
    ))
    
    header = "theta,blocking_prob,waiting_time,cpu_usage,ram_usage,effective_arrival_rate,waiting_requests,processing_requests"
    np.savetxt(
        f"{plots_dir}/theta_evaluation_results_{timestamp}.csv", 
        results, 
        delimiter=',', 
        header=header, 
        comments=''
    )
    
    # Collect metrics in a dictionary
    metrics_dict = {
        'theta_values': theta_values,
        'blocking_probs': blocking_probs,
        'waiting_times': waiting_times,
        'cpu_usages': cpu_usages,
        'ram_usages': ram_usages
    }
    
    # Find the best theta value
    best_theta, best_metrics, ranking = find_best_theta(metrics_dict)
    
    # Plot optimization results
    plot_optimization_results(theta_values, ranking, best_theta, plots_dir, timestamp)
    
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Best theta value: {best_theta:.4f}")
    print(f"Metrics at best theta:")
    print(f"  - Blocking probability: {best_metrics['blocking_prob']:.6f}")
    print(f"  - Waiting time: {best_metrics['waiting_time']:.6f}")
    print(f"  - CPU usage: {best_metrics['cpu_usage']:.6f}")
    print(f"  - RAM usage: {best_metrics['ram_usage']:.6f}")
    print(f"  - Overall score: {best_metrics['weighted_score']:.6f}")
    print("===========================\n")
    
    print(f"Evaluation complete. Results and plots saved in '{plots_dir}' directory.")
    return metrics_dict, best_theta, best_metrics

def evaluate_multiple_configs(n_iterations=1):
    """
    Evaluates the model across multiple system configurations and iterations
    
    Parameters:
    -----------
    n_iterations : int
        Number of iterations to run for each configuration to calculate confidence intervals
    
    Returns:
    --------
    all_results : DataFrame
        Pandas DataFrame containing all evaluation results
    """
    # Define parameter ranges to test
    param_ranges = {
        "lam": np.linspace(1.0, 10.0, 5),           # Arrival rate range from 1 to 10
        "mu": np.linspace(1.0, 10.0, 5),            # Service rate range from 1 to 10
        "alpha": np.linspace(0.1, 5.0, 5),          # Request activation rate range from 0.1 to 3.0
        "max_queue": np.linspace(5, 30, 5, dtype=int)  # Maximum queue size range from 5 to 20
    }
    
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
        theta_values = np.linspace(0.0, 1.0, 10)
        
        # For each theta value
        for theta_idx, theta in enumerate(theta_values):
            print(f"  Processing theta = {theta:.2f} ({theta_idx+1}/{len(theta_values)})")
            
            # Run multiple iterations to calculate confidence intervals
            iteration_results = []
            for iteration in range(n_iterations):
                # Create configuration for this run
                config = {
                    "lam": lam,
                    "mu": mu,
                    "alpha": alpha,
                    "beta": lam,  # Beta equals lambda as noted in the request
                    "theta": theta,
                    "max_queue": max_queue,
                    "ram_warm": 10,
                    "cpu_warm": 1,
                    "ram_demand": 25,
                    "cpu_demand": 30
                }
                
                # Run model and collect metrics
                model = MarkovModel(config, verbose=False)
                metrics = model.get_metrics()
                
                # Store metrics for this iteration
                iteration_results.append({
                    'lam': lam,
                    'mu': mu,
                    'alpha': alpha,
                    'beta': lam,
                    'theta': theta,
                    'max_queue': max_queue,
                    'iteration': iteration,
                    'blocking_prob': metrics["blocking_ratios"][0],
                    'waiting_time': metrics["waiting_times"][0],
                    'cpu_usage': metrics["mean_cpu_usage_per_request"][0],
                    'ram_usage': metrics["mean_ram_usage_per_request"][0],
                    'effective_arrival_rate': metrics["effective_arrival_rates"][0],
                    'waiting_requests': metrics["waiting_requests"][0],
                    'processing_requests': metrics["processing_requests"][0],
                    'idling_requests': metrics["idling_requests"][0]
                })
            
            # Add all iteration results to the main results list
            all_results.extend(iteration_results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save all raw results to CSV
    results_file = f"{results_dir}/optimization_results_raw_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Raw results saved to {results_file}")
    
    return results_df


def consolidate_and_plot_results(results_df=None, results_file=None, confidence_level=0.95, custom_weights=None):
    """
    Consolidates results across all configurations and creates a single plot 
    with confidence intervals representing variation across configurations
    
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
    """
    # Create directory for plots if it doesn't exist
    plots_dir = "optimization_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate timestamp for plot filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define default weights if not provided
    if custom_weights is None:
        custom_weights = {
            'blocking_prob': 0.4,    # Higher weight for blocking probability
            'waiting_time': 0.3,     # High weight for waiting time
            'cpu_usage': 0.15,       # Lower weight for CPU usage
            'ram_usage': 0.15        # Lower weight for RAM usage
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
        
        # Calculate mean and confidence intervals for each metric across all configurations
        consolidated_results = []
        
        for theta, group in grouped:
            # For each metric, calculate mean and confidence intervals
            metrics = ['blocking_prob', 'waiting_time', 'cpu_usage', 'ram_usage']
            result = {'theta': theta}
            
            for metric in metrics:
                values = group[metric].values
                mean = np.mean(values)
                # Calculate confidence interval
                if len(values) > 1:  # Need at least 2 samples for t-distribution
                    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=len(values)-1)
                    std_err = stats.sem(values)
                    margin = t_critical * std_err
                    lower_ci = mean - margin
                    upper_ci = mean + margin
                else:
                    lower_ci = mean
                    upper_ci = mean
                
                result[f'{metric}_mean'] = mean
                result[f'{metric}_lower'] = lower_ci
                result[f'{metric}_upper'] = upper_ci
            
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
    
    # Plot each metric with its confidence interval
    metrics_display = {
        'blocking_prob': ('Blocking Probability', 'blue'),
        'waiting_time': ('Waiting Time', 'red'),
        'cpu_usage': ('CPU Usage', 'green'),
        'ram_usage': ('RAM Usage', 'purple')
    }
    
    # Plot each normalized metric with its confidence interval
    for metric, (display_name, color) in metrics_display.items():
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
            
        # Plot mean line and confidence interval
        plt.plot(theta_values, norm_mean, '-', color=color, linewidth=fig_style["line_width"], label=display_name)
        plt.fill_between(theta_values, norm_lower, norm_upper, color=color, alpha=fig_style["alpha"])
    
    # Plot weighted score line
    plt.plot(theta_values, weighted_scores, 'k--', linewidth=fig_style["line_width"]+1, label='Weighted Score')
    
    # Mark the optimal theta
    # plt.axvline(x=best_theta, color='orange', linestyle='-', alpha=0.5, label=f'Optimal θ = {best_theta:.2f}')
    plt.plot(best_theta, best_score, 'ro', markersize=10)
    
    plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    plt.ylabel('Normalized Metric Value', fontsize=fig_style["label_fontsize"])
    # plt.title('Consolidated Metrics Across All Configurations', fontsize=fig_style["title_fontsize"])
    plt.xticks(fontsize=fig_style["tick_fontsize"])
    plt.yticks(fontsize=fig_style["tick_fontsize"])
    plt.xlim(0, 1)  # Set x-axis range from 0 to 1
    plt.ylim(-0.1, 1.1)  # Set y-axis range from 0 to 1
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fig_style["legend_fontsize"], bbox_to_anchor=(0.05, 0.8), loc='center left')

    
    # Save consolidated plot
    plot_file = f"{plots_dir}/consolidated_metrics_{timestamp}.png"
    # plt.savefig(plot_file, dpi=fig_style["dpi"], bbox_inches='tight')
    plt.show()
    
    print(f"Consolidated plot saved to {plot_file}")
    print(f"Optimal timeout rate (θ): {best_theta:.2f}")
    
    # Create individual metric plots too (optional)
    # for metric, (display_name, color) in metrics_display.items():
    #     plt.figure(figsize=fig_style["figure_size"])
        
    #     # Get data for this metric
    #     mean_values = cons_df[f'{metric}_mean'].values
    #     lower_values = cons_df[f'{metric}_lower'].values
    #     upper_values = cons_df[f'{metric}_upper'].values
        
    #     # Plot without normalization to see actual values
    #     plt.plot(theta_values, mean_values, '-', color=color, linewidth=fig_style["line_width"], label=display_name)
    #     plt.fill_between(theta_values, lower_values, upper_values, color=color, alpha=fig_style["alpha"])
        
    #     # Mark the optimal theta
    #     plt.axvline(x=best_theta, color='orange', linestyle='-', alpha=0.5, label=f'Optimal θ = {best_theta:.2f}')
        
    #     plt.xlabel('Timeout Rate (θ)', fontsize=fig_style["label_fontsize"])
    #     plt.ylabel(display_name, fontsize=fig_style["label_fontsize"])
    #     plt.title(f'{display_name} vs. Timeout Rate', fontsize=fig_style["title_fontsize"])
    #     plt.xticks(fontsize=fig_style["tick_fontsize"])
    #     plt.yticks(fontsize=fig_style["tick_fontsize"])
    #     plt.grid(True, alpha=0.3)
    #     plt.legend(fontsize=fig_style["legend_fontsize"])
        
    #     # Save individual plot
    #     # plt.savefig(f"{plots_dir}/consolidated_{metric}_{timestamp}.png", dpi=fig_style["dpi"], bbox_inches='tight')
    #     plt.show()
    
    print(f"Individual metric plots saved to {plots_dir} directory.")
    
    return cons_df, best_theta

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate Markov model with different configurations')
    parser.add_argument('--mode', choices=['full', 'plot_only'], default='full',
                      help='Run mode: full evaluation or plot from existing results')
    parser.add_argument('--results-file', type=str, 
                      help='Path to saved results file (for plot_only mode)')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of iterations for confidence interval calculation')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        print("Starting multi-configuration evaluation with confidence intervals...")
        # Run evaluations across different configurations
        results_df = evaluate_multiple_configs(n_iterations=args.iterations)
        # Consolidate and plot results
        cons_df, best_theta = consolidate_and_plot_results(results_df=results_df, confidence_level=0.95)
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
            custom_weights=custom_weights
        )
        print(f"Plotting complete! Optimal theta: {best_theta:.4f}")