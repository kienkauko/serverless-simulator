#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for the 4D Markov model with varying timeout rate (theta) and model decay rate (gamma)
This script analyzes how different values of theta and gamma affect various metrics
across system configurations
"""

import matplotlib.pyplot as plt
import numpy as np
from model_4D import MarkovModel
import os
import datetime
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def find_best_parameters(metrics_dict, weights=None):
    """
    Find the best theta and gamma values that optimize multiple metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing parameter values and metrics arrays
    weights : dict, optional
        Dictionary of weights for each metric. Default weights prioritize
        blocking probability and waiting time over resource usage.
    
    Returns:
    --------
    best_params : dict
        The theta and gamma values that yield the optimal combined performance
    best_metrics : dict
        Dictionary of metrics at the best parameter values
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
    gamma_values = metrics_dict['gamma_values']
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
    best_idx = np.unravel_index(np.argmin(weighted_scores), weighted_scores.shape)
    best_theta = theta_values[best_idx[0]]
    best_gamma = gamma_values[best_idx[1]]
    
    # Create results dictionary
    best_metrics = {
        'theta': best_theta,
        'gamma': best_gamma,
        'blocking_prob': blocking_probs[best_idx],
        'waiting_time': waiting_times[best_idx],
        'cpu_usage': cpu_usages[best_idx],
        'ram_usage': ram_usages[best_idx],
        'weighted_score': weighted_scores[best_idx]
    }
    
    # Create ranking dictionary with normalized scores
    ranking = {
        'theta_values': theta_values,
        'gamma_values': gamma_values, 
        'norm_blocking': norm_blocking,
        'norm_waiting': norm_waiting,
        'norm_cpu': norm_cpu,
        'norm_ram': norm_ram,
        'weighted_scores': weighted_scores
    }
    
    return best_metrics, ranking

def plot_3d_metric(theta_values, gamma_values, metric_values, metric_name, best_theta=None, best_gamma=None):
    """
    Plot a 3D surface of a metric as a function of theta and gamma
    
    Parameters:
    -----------
    theta_values : array
        Array of theta values that were evaluated
    gamma_values : array
        Array of gamma values that were evaluated
    metric_values : 2D array
        2D array of metric values for each (theta, gamma) pair
    metric_name : str
        Name of the metric being plotted
    best_theta : float, optional
        The best theta value identified
    best_gamma : float, optional
        The best gamma value identified
    """
    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mesh grid for plotting
    Theta, Gamma = np.meshgrid(theta_values, gamma_values, indexing='ij')
    
    # Plot the surface
    surf = ax.plot_surface(Theta, Gamma, metric_values, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Timeout Rate (θ)', fontsize=14)
    ax.set_ylabel('Model Decay Rate (γ)', fontsize=14)
    ax.set_zlabel(metric_name, fontsize=14)
    ax.set_title(f'Impact of θ and γ on {metric_name}', fontsize=16)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Mark the best point if provided
    if best_theta is not None and best_gamma is not None:
        # Find the closest indices for the best parameter values
        theta_idx = np.abs(theta_values - best_theta).argmin()
        gamma_idx = np.abs(gamma_values - best_gamma).argmin()
        
        # Get the corresponding metric value
        z_value = metric_values[theta_idx, gamma_idx]
        
        # Plot the best point as a red dot
        ax.scatter([best_theta], [best_gamma], [z_value], color='red', s=100, label=f'Best Point (θ={best_theta:.2f}, γ={best_gamma:.2f})')
        ax.legend()

def evaluate_theta_gamma_impact():
    """
    Evaluates the impact of varying theta and gamma on different metrics
    Generates 3D plots showing the relationship between (theta, gamma) and:
    - Blocking probability
    - Waiting time
    - CPU usage
    - RAM usage
    - Weighted score (combined metric)
    
    Returns:
    --------
    metrics_dict : dict
        Dictionary containing metrics data
    best_metrics : dict
        Dictionary with metrics at the optimal parameter values
    """
    print("Evaluating model with different theta and gamma values...")
    
    # Define base configuration with fixed values
    base_config = {
        "lam": 5,          # Arrival rate
        "mu": 5,           # Service rate
        "alpha": 1/8,      # Request activation rate
        "beta": 5,         # Idle encounter rate
        "theta": 0.0,      # Container timeout rate (to be varied)
        "gamma": 0.0,      # Model decay rate (to be varied)
        "sigma": 1/4,        # Fixed cold to warm transition rate
        "max_queue": 5,    # Maximum queue size
        "ram_warm_cpu": 10,    # RAM usage in warm state
        "ram_warm_model": 20,    # RAM usage in warm state
        "cpu_warm_cpu": 1,     # CPU usage in warm state
        "cpu_warm_model": 1,     # CPU usage in warm state
        "ram_demand": 30,  # RAM usage in active state
        "cpu_demand": 30   # CPU usage in active state
    }
    
    # Define ranges of parameter values to evaluate
    theta_values = np.linspace(0.1, 2.0, 8)   # From 0.1 to 2.0 with 8 points
    gamma_values = np.linspace(0.1, 2.0, 8)   # From 0.1 to 2.0 with 8 points
    
    # Initialize storage for metrics as 2D arrays
    blocking_probs = np.zeros((len(theta_values), len(gamma_values)))
    waiting_times = np.zeros((len(theta_values), len(gamma_values)))
    cpu_usages = np.zeros((len(theta_values), len(gamma_values)))
    ram_usages = np.zeros((len(theta_values), len(gamma_values)))
    
    # Evaluate model for each theta-gamma combination
    for i, theta in enumerate(theta_values):
        for j, gamma in enumerate(gamma_values):
            print(f"Processing theta = {theta:.2f}, gamma = {gamma:.2f}")
            
            # Update config with current parameter values
            config = base_config.copy()
            config["theta"] = theta
            config["gamma"] = gamma
            
            # Create and evaluate model
            model = MarkovModel(config, verbose=False)
            metrics = model.get_metrics()
            
            # Store metrics
            blocking_probs[i, j] = metrics["blocking_ratios"][0]
            waiting_times[i, j] = metrics["waiting_times"][0]
            cpu_usages[i, j] = metrics["mean_cpu_usage_per_request"][0]
            ram_usages[i, j] = metrics["mean_ram_usage_per_request"][0]
    
    # Collect metrics in a dictionary
    metrics_dict = {
        'theta_values': theta_values,
        'gamma_values': gamma_values,
        'blocking_probs': blocking_probs,
        'waiting_times': waiting_times,
        'cpu_usages': cpu_usages,
        'ram_usages': ram_usages
    }
    
    # Find the best parameter values
    best_metrics, ranking = find_best_parameters(metrics_dict)
    
    # Plot each metric against theta and gamma
    plot_3d_metric(theta_values, gamma_values, blocking_probs, "Blocking Probability", 
                  best_metrics['theta'], best_metrics['gamma'])
    
    plot_3d_metric(theta_values, gamma_values, waiting_times, "Waiting Time", 
                  best_metrics['theta'], best_metrics['gamma'])
    
    plot_3d_metric(theta_values, gamma_values, cpu_usages, "CPU Usage per Request", 
                  best_metrics['theta'], best_metrics['gamma'])
    
    plot_3d_metric(theta_values, gamma_values, ram_usages, "RAM Usage per Request", 
                  best_metrics['theta'], best_metrics['gamma'])
    
    plot_3d_metric(theta_values, gamma_values, ranking['weighted_scores'], "Weighted Score (lower is better)", 
                  best_metrics['theta'], best_metrics['gamma'])
    
    # Print optimization results
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Best theta value: {best_metrics['theta']:.4f}")
    print(f"Best gamma value: {best_metrics['gamma']:.4f}")
    print(f"Metrics at best parameters:")
    print(f"  - Blocking probability: {best_metrics['blocking_prob']:.6f}")
    print(f"  - Waiting time: {best_metrics['waiting_time']:.6f}")
    print(f"  - CPU usage: {best_metrics['cpu_usage']:.6f}")
    print(f"  - RAM usage: {best_metrics['ram_usage']:.6f}")
    print(f"  - Overall score: {best_metrics['weighted_score']:.6f}")
    print("===========================\n")
    
    return metrics_dict, best_metrics

def plot_heatmaps(metrics_dict, best_metrics):
    """
    Plot 2D heatmaps for each metric with theta and gamma on the axes
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary containing metrics data
    best_metrics : dict
        Dictionary with metrics at the optimal parameter values
    """
    theta_values = metrics_dict['theta_values']
    gamma_values = metrics_dict['gamma_values']
    
    # Set up a 2x3 grid of subplots for all metrics and the combined score
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = [
        ('blocking_probs', 'Blocking Probability'),
        ('waiting_times', 'Waiting Time'),
        ('cpu_usages', 'CPU Usage'),
        ('ram_usages', 'RAM Usage'),
        ('weighted_scores', 'Weighted Score')
    ]
    
    # Extract best parameter values
    best_theta = best_metrics['theta']
    best_gamma = best_metrics['gamma']
    
    # Find the nearest indices for the best parameters
    theta_idx = np.abs(theta_values - best_theta).argmin()
    gamma_idx = np.abs(gamma_values - best_gamma).argmin()
    
    # Plot each metric as a heatmap
    for i, (metric_key, metric_name) in enumerate(metrics):
        # Skip the last subplot
        if i >= len(axes) - 1:
            break
            
        ax = axes[i]
        
        # Get the appropriate data
        if metric_key == 'weighted_scores':
            # Need to compute this from the normalized metrics
            norm_blocking = metrics_dict['norm_blocking']
            norm_waiting = metrics_dict['norm_waiting']
            norm_cpu = metrics_dict['norm_cpu']
            norm_ram = metrics_dict['norm_ram']
            
            # Default weights
            weights = {
                'blocking_probs': 0.4,
                'waiting_times': 0.3,
                'cpu_usages': 0.15,
                'ram_usages': 0.15
            }
            
            data = (weights['blocking_probs'] * norm_blocking +
                   weights['waiting_times'] * norm_waiting +
                   weights['cpu_usages'] * norm_cpu +
                   weights['ram_usages'] * norm_ram)
        else:
            data = metrics_dict[metric_key]
        
        # Create heatmap
        im = ax.imshow(data.T, origin='lower', aspect='auto', cmap='viridis',
                      extent=[min(theta_values), max(theta_values), min(gamma_values), max(gamma_values)])
        
        # Mark the best point
        ax.plot(best_theta, best_gamma, 'r*', markersize=10)
        
        # Add labels and title
        ax.set_xlabel('Timeout Rate (θ)')
        ax.set_ylabel('Model Decay Rate (γ)')
        ax.set_title(f"{metric_name}")
        
        # Add colorbar
        fig.colorbar(im, ax=ax)
    
    # Use the last subplot for a summary or remove it
    axes[-1].axis('off')
    
    # Add a title for the entire figure
    plt.suptitle("Impact of Timeout Rate (θ) and Model Decay Rate (γ) on System Metrics", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

if __name__ == "__main__":
    # Evaluate theta and gamma impact
    metrics_dict, best_metrics = evaluate_theta_gamma_impact()
    
    # Display plots (main visualization, 3D plots are created in evaluate_theta_gamma_impact)
    plt.show()