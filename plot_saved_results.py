#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting script to visualize saved lambda-theta relationship results
without rerunning the simulations
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse

def plot_saved_lambda_theta_relationship(csv_file):
    """
    Plot the lambda-theta relationship from a saved CSV file with confidence intervals
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing lambda-theta relationship data
    """
    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return
    
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    lambda_theta_df = pd.read_csv(csv_file)
    
    # Print column names to verify data structure
    print("Available columns:", lambda_theta_df.columns.tolist())
    
    # Create directory for plots if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
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
    
    # Check if all required columns exist and extract data
    required_columns = [
        'best_theta_overall', 'best_theta_blocking', 'best_theta_waiting', 
        'best_theta_cpu', 'best_theta_ram',
        'best_overall_error'  # Standard deviation of overall best theta
    ]
    
    for col in required_columns:
        if col not in lambda_theta_df.columns:
            print(f"Warning: Column '{col}' not found in the CSV file.")
    
    # Extract data (with fallbacks for missing columns)
    best_thetas_overall = lambda_theta_df['best_theta_overall'].values if 'best_theta_overall' in lambda_theta_df.columns else None
    best_thetas_blocking = lambda_theta_df['best_theta_blocking'].values if 'best_theta_blocking' in lambda_theta_df.columns else None
    best_thetas_waiting = lambda_theta_df['best_theta_waiting'].values if 'best_theta_waiting' in lambda_theta_df.columns else None
    best_thetas_cpu = lambda_theta_df['best_theta_cpu'].values if 'best_theta_cpu' in lambda_theta_df.columns else None
    best_thetas_ram = lambda_theta_df['best_theta_ram'].values if 'best_theta_ram' in lambda_theta_df.columns else None
    
    # Calculate 95% confidence interval multiplier (using t-distribution with 4 degrees of freedom for 5 samples)
    # For 95% CI with 4 degrees of freedom (5 samples), t-value is approximately 2.776
    t_value = 2.776
    
    # Create figure for combined plot of all metrics with confidence intervals
    plt.figure(figsize=fig_style["figure_size"])
    
    # Plot best theta for each metric with confidence bands
    # For overall theta - using a slightly transparent confidence band
    if best_thetas_overall is not None:
        plt.plot(lambda_values, best_thetas_overall, 'k-o', linewidth=fig_style["line_width"], 
                markersize=fig_style["marker_size"], label='Overall Best θ')
        
        # Add confidence interval for overall theta if available
        # This uses the directly calculated standard deviation of best theta values
        if 'best_overall_error' in lambda_theta_df.columns:
            overall_errors = lambda_theta_df['best_overall_error'].values * t_value
            plt.fill_between(lambda_values, 
                            best_thetas_overall - overall_errors, 
                            best_thetas_overall + overall_errors, 
                            color='k', alpha=0.1)
    
    # For blocking probability theta
    if best_thetas_blocking is not None:
        plt.plot(lambda_values, best_thetas_blocking, 'b-s', linewidth=fig_style["line_width"], 
                markersize=fig_style["marker_size"], label='Best θ for Blocking Prob')
    
    # For waiting time theta
    if best_thetas_waiting is not None:
        plt.plot(lambda_values, best_thetas_waiting, 'r-^', linewidth=fig_style["line_width"], 
                markersize=fig_style["marker_size"], label='Best θ for Waiting Time')
    
    # For CPU usage theta
    if best_thetas_cpu is not None:
        plt.plot(lambda_values, best_thetas_cpu, 'g-d', linewidth=fig_style["line_width"], 
                markersize=fig_style["marker_size"], label='Best θ for CPU Usage')
    
    # For RAM usage theta
    if best_thetas_ram is not None:
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
    
    # Show the plot
    plt.show()
    
    print("Lambda-theta relationship plot displayed successfully")

def plot_saved_metrics_with_ci(csv_file):
    """
    Plot the individual metrics at best theta from a saved CSV file with confidence intervals
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing lambda-theta relationship data
    """
    # Check if the file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return
    
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    lambda_theta_df = pd.read_csv(csv_file)
    
    # Create directory for plots if it doesn't exist
    plots_dir = "theta_evaluation_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
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
    
    # Check if metric columns exist
    metrics_columns = [
        'best_blocking_prob', 'best_waiting_time', 'best_cpu_usage', 'best_ram_usage',
        'best_blocking_error', 'best_waiting_error', 'best_cpu_error', 'best_ram_error'
    ]
    
    for col in metrics_columns:
        if col not in lambda_theta_df.columns:
            print(f"Warning: Column '{col}' not found in the CSV file.")
    
    # Create subplot for all metrics with confidence intervals
    fig, axs = plt.subplots(2, 2, figsize=(fig_style["figure_size"][0], fig_style["figure_size"][1]))
    fig.suptitle('Performance Metrics with 95% Confidence Intervals', fontsize=fig_style["title_fontsize"])
    
    # Plot blocking probability with confidence interval
    if all(col in lambda_theta_df.columns for col in ['best_blocking_prob', 'best_blocking_error']):
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
    if all(col in lambda_theta_df.columns for col in ['best_waiting_time', 'best_waiting_error']):
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
    if all(col in lambda_theta_df.columns for col in ['best_cpu_usage', 'best_cpu_error']):
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
    if all(col in lambda_theta_df.columns for col in ['best_ram_usage', 'best_ram_error']):
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
    
    print("Metrics plot with confidence intervals displayed successfully")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Plot lambda-theta relationship from saved CSV file')
    parser.add_argument('--file', type=str, default='optimization_results/lambda_theta_relationship_20250515_152535.csv',
                      help='Path to the CSV file containing lambda-theta relationship data')
    parser.add_argument('--plot-type', type=str, choices=['theta', 'metrics', 'both'], default='theta',
                      help='Type of plot to generate: theta, metrics, or both')
    
    args = parser.parse_args()
    
    if args.plot_type in ['theta', 'both']:
        plot_saved_lambda_theta_relationship(args.file)
    
    if args.plot_type in ['metrics', 'both']:
        plot_saved_metrics_with_ci(args.file)