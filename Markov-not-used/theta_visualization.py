#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 2025

Visualization functions for theta optimization study results.
This module provides functions to visualize results from CSV files
without having to rerun the optimization calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def load_profile_results_from_csv(csv_path):
    """
    Load profile results from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing the loaded profile results
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Extract unique profile names
    profile_names = df['profile_name'].unique()
    
    # Dictionary to store results
    all_profile_results = {}
    
    # Metrics to extract
    metric_names = ["blocking_ratios", "latency", "ram_usage", "cpu_usage"]
    
    # Process each profile
    for profile_name in profile_names:
        # Filter data for this profile
        profile_data = df[df['profile_name'] == profile_name]
        
        # Create results dictionary for this profile
        results = {
            "lambda_values": profile_data['lambda'].values,
            "profile_name": profile_name,
            "profile_color": profile_data['profile_color'].iloc[0],
            "profile_marker": profile_data['profile_marker'].iloc[0]
        }
        
        # Extract metrics
        for metric in metric_names:
            results[f"best_theta_{metric}"] = profile_data[f'best_theta_{metric}'].values
            results[f"best_{metric}_value"] = profile_data[f'best_{metric}_value'].values
        
        # Store in all_profile_results
        all_profile_results[profile_name] = results
    
    return all_profile_results

def visualize_metrics_from_csv(csv_path):
    """
    Visualize metrics from a CSV file containing saved results.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    # Load results from CSV
    all_profile_results = load_profile_results_from_csv(csv_path)
    
    # Extract profile parameters for reference lines
    profiles = []
    for profile_name, results in all_profile_results.items():
        # Get first row of data to extract parameters
        df = pd.read_csv(csv_path)
        profile_data = df[df['profile_name'] == profile_name].iloc[0]
        
        profile = {
            "name": profile_name,
            "color": results["profile_color"],
            "marker": results["profile_marker"],
            "service_rate": profile_data['service_rate'],
            "spawn_rate": profile_data['spawn_rate']
        }
        profiles.append(profile)
    
    # Metrics to analyze individually
    metric_names = ["blocking_ratios", "latency", "ram_usage", "cpu_usage"]
    metric_display_names = ["Blocking Probability", "Latency", "RAM Usage", "CPU Usage"]
    
    # Create plots - one figure showing all metrics across profiles
    plt.figure(figsize=(14, 10))
    
    # Create 2x2 subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot best theta for each metric, one subplot per metric
    for i, metric in enumerate(metric_names):
        for profile_name, results in all_profile_results.items():
            axes[i].plot(results["lambda_values"], results[f"best_theta_{metric}"], 
                      marker=results["profile_marker"], linestyle='-', 
                      linewidth=2, markersize=6, 
                      color=results["profile_color"], 
                      label=f"{profile_name}")
        
        axes[i].set_xlabel('Lambda (Arrival Rate)', fontsize=12)
        axes[i].set_ylabel('Best Theta Value', fontsize=12)
        axes[i].set_title(f'Best Theta for {metric_display_names[i]}', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)
    
    plt.tight_layout()
    plt.suptitle(f'Best Theta Values by Metric and Profile', fontsize=16, y=1.02)
    plt.show()
    
    # Create individual plots - one figure for each metric showing all profiles
    for i, metric in enumerate(metric_names):
        plt.figure(figsize=(12, 8))
        
        for profile_name, results in all_profile_results.items():
            plt.plot(results["lambda_values"], results[f"best_theta_{metric}"], 
                   marker=results["profile_marker"], linestyle='-', 
                   linewidth=2, markersize=8, 
                   color=results["profile_color"], 
                   label=f"{profile_name}")
        
        plt.xlabel('Lambda (Arrival Rate)', fontsize=12)
        plt.ylabel('Best Theta Value', fontsize=12)
        plt.title(f'Best Theta Values for {metric_display_names[i]} vs Lambda', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        
    # Create individual plots showing the metric values achieved with optimal theta
    for i, metric in enumerate(metric_names):
        plt.figure(figsize=(12, 8))
        
        # Plot achieved metric values for each profile
        for profile_name, results in all_profile_results.items():
            profile_params = next(p for p in profiles if p["name"] == profile_name)
            
            plt.plot(results["lambda_values"], results[f"best_{metric}_value"], 
                   marker=results["profile_marker"], linestyle='-', 
                   linewidth=2, markersize=8, 
                   color=results["profile_color"], 
                   label=f"{profile_name}")
            
        plt.xlabel('Lambda (Arrival Rate)', fontsize=12)
        plt.ylabel(f'{metric_display_names[i]} Value', fontsize=12)
        plt.title(f'Best {metric_display_names[i]} Values vs Lambda', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add reference lines or additional information if needed
        if metric == "blocking_ratios":
            plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.3)
            plt.text(results["lambda_values"][0], 0.052, "5% blocking threshold", fontsize=10)
        
        # For latency, add theoretical min latency line for each profile
        elif metric == "latency":
            for profile in profiles:
                min_latency = 1/profile["service_rate"]
                plt.axhline(y=min_latency, color=profile["color"], linestyle='--', alpha=0.3)
                plt.text(results["lambda_values"][0], min_latency*1.05, 
                        f"Min latency {profile['name']} (1/μ = {min_latency:.2f})", 
                        fontsize=10, color=profile["color"])
        
        plt.tight_layout()
        plt.show()
    
    # Create 2x2 subplot figure showing all metric values achieved
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_names):
        for profile_name, results in all_profile_results.items():
            axes[i].plot(results["lambda_values"], results[f"best_{metric}_value"],
                      marker=results["profile_marker"], linestyle='-', 
                      linewidth=2, markersize=6,
                      color=results["profile_color"],
                      label=f"{profile_name}")
            
        axes[i].set_xlabel('Lambda (Arrival Rate)', fontsize=12)
        axes[i].set_ylabel(f'{metric_display_names[i]}', fontsize=12)
        axes[i].set_title(f'{metric_display_names[i]} vs Lambda', fontsize=14)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)
        
        # Add reference lines if applicable
        if metric == "blocking_ratios":
            axes[i].axhline(y=0.05, color='r', linestyle='--', alpha=0.3)
        
        elif metric == "latency":
            for profile in profiles:
                min_latency = 1/profile["service_rate"]
                axes[i].axhline(y=min_latency, color=profile["color"], linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Metric Values with Optimal Theta Settings', fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("This module provides functions to visualize results from saved CSV files.")
    print("Example usage:")
    print("  visualize_metrics_from_csv('path/to/saved_results.csv')")
    
    # Check if there are any CSV files in the optimization_results directory
    optimization_dir = "../optimization_results"
    if os.path.exists(optimization_dir):
        csv_files = [f for f in os.listdir(optimization_dir) if f.endswith('.csv') and 'profile_results' in f]
        if csv_files:
            print("\nFound the following result files:")
            for i, csv_file in enumerate(csv_files):
                print(f"  {i+1}. {csv_file}")
            print("\nYou can visualize them using:")
            print(f"  visualize_metrics_from_csv('{optimization_dir}/{csv_files[-1]}')")
