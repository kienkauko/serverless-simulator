#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization tool for theta-gamma-lambda results
Study of how both theta and gamma parameters affect system metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
from datetime import datetime
import multiprocessing as mp
from functools import partial
import math

# Import scipy for confidence interval calculation
from scipy import stats
from model_4D import MarkovModel
import simpy
from System import System
from Server import Server

def run_simulator(sim_config):
    """Run the simulator and extract metrics"""
    # Create simulation environment
    env = simpy.Environment()
    
    # Create system
    system = System(env, sim_config, distribution=sim_config["system"]["distribution"], 
                    verbose=sim_config["system"]["verbose"])
    
    # Add servers
    for i in range(sim_config["system"]["num_servers"]):
        server = Server(env, f"Server-{i}", sim_config["server"])
        system.add_server(server)
    
    env.process(system.request_generator())
    # env.process(system.resource_monitor_process())
    
    # Run simulation
    env.run(until=sim_config["system"]["sim_time"])

    return system.get_metrics()

def convert_input(config):
    """
    Convert Markov model configuration to simulation configuration format.
    
    Args:
        config (dict): Markov model configuration
        
    Returns:
        dict: Simulation configuration
    """
    sim_config = {
        # System Parameters
        "system": {
            "num_servers": config["num_servers"],  # Default value, could be made configurable
            "sim_time": 10000,  # Longer simulation for stability
            "distribution": "exponential",  # Use exponential distribution for request arrivals
            "verbose": False
        },
        
        # Server Parameters
        "server": {
            "cpu_capacity": 100.0,
            "ram_capacity": 100.0,
            "power_max": config["power_max"],  # Maximum power consumption
            "power_min_scale": config["power_min_scale"],  # Scale for minimum power consumption
        
        },
        
        # Request Parameters
        "request": {
            "arrival_rate": config["arrival_rate"],
            "service_rate": config["service_rate"],
            "warm_cpu": config["cpu_warm"],
            "warm_ram": config["ram_warm"],
            "warm_cpu_model": config["cpu_warm_model"],
            "warm_ram_model": config["ram_warm_model"],
            "cpu_demand": config["cpu_demand"],
            "ram_demand": config["ram_demand"],
        },
        
        # Container Parameters
        "container": {
            "spawn_time": 1/config["spawn_rate"],  # Convert spawn_rate to spawn_time
            "idle_cpu_timeout": 1/config["theta"],  # Convert theta to timeout
            "idle_model_timeout": 1/config["gamma"],  # Small value to avoid immediate removal
            "load_request_time": 1/10000,
            "load_model_time": 1/config["spawn_model_rate"],
        },
    }
    
    return sim_config

def process_theta(lambda_val, theta, profile_params, use_simulation):
    """
    Worker function to process a specific lambda-theta combination.
    
    Args:
        lambda_val: The arrival rate value
        theta: The theta value to test
        profile_params: System parameters
        use_simulation: Whether to use simulation or Markov model
        
    Returns:
        tuple: (theta, metrics_dict) where metrics_dict contains results for each metric
    """
    # Create config with current lambda and theta
    config = profile_params.copy()
    config["arrival_rate"] = lambda_val
    config["theta"] = theta
    
    # Run the Markov model or simulation
    if not use_simulation:
        model = MarkovModel(config, verbose=False)
        metrics = model.get_metrics()
    else:
        sim_config = convert_input(config)
        metrics = run_simulator(sim_config)
    
    # Return results
    return theta, metrics


def visualize_fixed_lambda_results(results_path=None, results=None, confidence=0.95, plot_type='2D'):
    """
    Create visualizations for the fixed lambda results, showing how metrics
    vary with theta and gamma for each profile.
    
    Args:
        results_path (str): Path to the CSV file with results
        results (dict): Results dictionary (alternative to loading from CSV)
        confidence (float): Confidence level for interval calculation (default: 0.95)
        plot_type (str): Type of plot to create - '2D', '3D', or 'both' (default: '2D')
    
    Returns:
        None
    """
    # Define font sizes for consistent styling
    TITLE_FONTSIZE = 16
    AXIS_LABEL_FONTSIZE = 22
    TICK_FONTSIZE = 22
    LEGEND_FONTSIZE = 12
    SUPTITLE_FONTSIZE = 18
    ANNOTATION_FONTSIZE = 18
    
    
    # Load results from CSV if path is provided
    if results_path and results is None:
        df = pd.read_csv(results_path)
        
        # Get unique profiles
        profiles = []
        for profile_name in df['profile_name'].unique():
            profile_data = df[df['profile_name'] == profile_name].iloc[0]
            profiles.append({
                "name": profile_name,
                "color": profile_data['profile_color'],
                "marker": profile_data['profile_marker']
            })
        
        # Check if gamma column exists (2D study)
        has_gamma = 'gamma' in df.columns
        print(f"Gamma values found: {has_gamma}")
        # Reconstruct results dictionary
        results = {}
        for profile in profiles:
            profile_df = df[df['profile_name'] == profile['name']]
            
            # Get unique theta and gamma values (if available)
            theta_values = sorted(profile_df['theta'].unique())
            gamma_values = sorted(profile_df['gamma'].unique()) if has_gamma else [None]
            
            profile_results = {
                "theta_values": theta_values,
                "lambda": profile_df['lambda'].iloc[0],
                "profile_name": profile['name'],
                "profile_color": profile['color'],
                "profile_marker": profile['marker']
            }
            
            if has_gamma:
                profile_results["gamma_values"] = gamma_values
                print("been here")
                # For 2D parameter sweep, create 2D grids for each metric
                metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
                
                for metric in metric_names:
                    if metric in profile_df.columns:
                        # Create a 2D matrix to store the metric values
                        metric_grid = np.full((len(theta_values), len(gamma_values)), np.nan)
                        
                        for theta_idx, theta in enumerate(theta_values):
                            for gamma_idx, gamma in enumerate(gamma_values):
                                subset = profile_df[(profile_df['theta'] == theta) & (profile_df['gamma'] == gamma)]
                                if not subset.empty and metric in subset.columns:
                                    metric_grid[theta_idx, gamma_idx] = subset[metric].mean()
                        
                        profile_results[metric] = metric_grid
            else:
                # Handle traditional 1D metrics (theta only)
                metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
                
                for metric in metric_names:
                    if metric in profile_df.columns:
                        # Store mean values for each theta
                        avg_values = []
                        
                        for theta in theta_values:
                            theta_df = profile_df[profile_df['theta'] == theta]
                            if not theta_df.empty:
                                avg_values.append(theta_df[metric].mean())
                            else:
                                avg_values.append(None)
                                
                        profile_results[metric] = avg_values
            
            results[profile['name']] = profile_results
    
    if not results:
        print("No results provided or loaded.")
        return
    
    # Check if results include gamma values (2D parameter sweep)
    has_gamma = all("gamma_values" in results[profile_name] for profile_name in results)
    print(f"Gamma values found in results: {has_gamma}")
    # Metric information for plotting
    metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
    metric_display_names = ["Blocking Probability", "Latency (s)", "RAM Usage Per Request (%)", "CPU Usage Per Request (%)", "Power Usage Per Request (W)"]
    print("plot type:", plot_type)
    # Create plots based on the requested plot type
    if plot_type in ['2D', 'both'] and has_gamma:
        # Create 2D plots (theta vs metric) for selected gamma values
        print("\nCreating 2D plots...")
        
        # Get a sample profile for gamma values
        sample_profile = next(iter(results.values()))
        gamma_values = sample_profile.get("gamma_values", [0.001])
        
        # For each gamma value, create a set of 2D plots
        for gamma_idx, gamma in enumerate(gamma_values):
            if gamma_idx > 2 and gamma_idx < len(gamma_values) - 1:
                # Skip middle gamma values if there are many
                continue
                
            print(f"Creating 2D plots for gamma = {gamma:.6f}")
            
            # Create individual plots for each metric
            for i, metric in enumerate(metric_names):
                plt.figure(figsize=(6, 4.8))
                
                # Dictionary to store min values and corresponding thetas for each profile
                min_values = {}
                
                # Plot each profile
                for profile_name, profile_results in results.items():
                    if metric in profile_results:
                        # Extract values for this gamma value
                        theta_values = profile_results["theta_values"]
                        
                        if isinstance(profile_results[metric], np.ndarray):
                            # Data is in 2D grid, extract the row for this gamma
                            metric_values = profile_results[metric][:, gamma_idx]
                        else:
                            # Data is already a 1D array
                            metric_values = profile_results[metric]
                        
                        # Plot mean line
                        line, = plt.plot(theta_values, metric_values, 
                               marker=profile_results["profile_marker"], linestyle='-', 
                               linewidth=3, markersize=6, 
                               color=profile_results["profile_color"], 
                               label=f"{profile_name}")
                        
                        # Find the minimum value and corresponding theta
                        valid_values = [(theta, val) for theta, val in zip(theta_values, metric_values) 
                                       if val is not None and not np.isnan(val)]
                        
                        if valid_values:
                            thetas, vals = zip(*valid_values)
                            min_idx = np.argmin(vals)
                            min_theta = thetas[min_idx]
                            min_value = vals[min_idx]
                            min_values[profile_name] = (min_theta, min_value)
                            
                            # Add vertical line at minimum value
                            plt.axvline(x=min_theta, color="black", 
                                        linestyle='--', alpha=0.7, linewidth=2)
                            
                            # Add annotation for optimal theta value
                            plt.annotate(f'θ={min_theta:.2f}',
                                        xy=(min_theta, min_value),
                                        xytext=(10, 150),  # Offset text
                                        textcoords='offset points',
                                        fontsize=ANNOTATION_FONTSIZE,
                                        color="black",
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
                            
                plt.xlabel('Theta Value', fontsize=AXIS_LABEL_FONTSIZE)        
                plt.ylabel(metric_display_names[i], fontsize=AXIS_LABEL_FONTSIZE)
                plt.title(f'γ = {gamma:.6f}', fontsize=TITLE_FONTSIZE)
                plt.grid(True, alpha=0.3)
                plt.xticks(fontsize=TICK_FONTSIZE)
                plt.yticks(fontsize=TICK_FONTSIZE)
                
                plt.tight_layout()
                plt.show()
    
    if plot_type in ['3D', 'both'] and has_gamma:
        # Create 3D plots (theta vs gamma vs metric)
        print("\nCreating 3D plots...")
        
        for i, metric in enumerate(metric_names):
            for profile_name, profile_results in results.items():
                if metric in profile_results:
                    # Create the 3D figure
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Get the parameter grids
                    theta_values = profile_results["theta_values"]
                    gamma_values = profile_results["gamma_values"]
                    
                    # Create meshgrid for 3D plotting
                    THETA, GAMMA = np.meshgrid(theta_values, gamma_values, indexing='ij')
                    
                    # Get metric values
                    if isinstance(profile_results[metric], np.ndarray):
                        Z = profile_results[metric]
                    else:
                        # If data is not in 2D array format, convert it
                        Z = np.array(profile_results[metric]).reshape(len(theta_values), -1)
                    
                    # Create surface plot
                    surf = ax.plot_surface(THETA, GAMMA, Z, cmap='viridis', 
                                          edgecolor='none', alpha=0.8)
                    
                    # Add color bar
                    cbar = fig.colorbar(surf, ax=ax, pad=0.1)
                    cbar.ax.set_ylabel(metric_display_names[i], fontsize=AXIS_LABEL_FONTSIZE-8)
                    
                    # Find the minimum value point
                    min_idx = np.nanargmin(Z)
                    min_theta_idx, min_gamma_idx = np.unravel_index(min_idx, Z.shape)
                    min_theta = theta_values[min_theta_idx]
                    min_gamma = gamma_values[min_gamma_idx]
                    min_value = Z[min_theta_idx, min_gamma_idx]
                    
                    # Add a point to mark the minimum
                    ax.scatter([min_theta], [min_gamma], [min_value], 
                               color='red', s=100, marker='*', 
                               label=f'Min: θ={min_theta:.2f}, γ={min_gamma:.6f}, {metric_display_names[i]}={min_value:.2f}')
                    
                    # Set labels
                    ax.set_xlabel('Theta (θ)', fontsize=AXIS_LABEL_FONTSIZE-8)
                    ax.set_ylabel('Gamma (γ)', fontsize=AXIS_LABEL_FONTSIZE-8)
                    ax.set_zlabel(metric_display_names[i], fontsize=AXIS_LABEL_FONTSIZE-8)
                    ax.set_title(f'{profile_name} - {metric_display_names[i]} vs. Theta and Gamma', fontsize=TITLE_FONTSIZE)
                    
                    # Add legend
                    ax.legend(fontsize=LEGEND_FONTSIZE)
                    
                    plt.tight_layout()
                    plt.show()
    
    elif plot_type in ['2D', 'both'] and not has_gamma:
        # Create traditional 2D plots (theta vs metric)
        print("\nCreating traditional 2D plots (theta only)...")
        
        # Create individual plots for each metric
        for i, metric in enumerate(metric_names):
            plt.figure(figsize=(6, 4.8))
            
            # Dictionary to store min values and corresponding thetas for each profile
            min_values = {}
            
            # Plot each profile
            for profile_name, profile_results in results.items():
                if metric in profile_results:
                    # Plot mean line
                    line, = plt.plot(profile_results["theta_values"], profile_results[metric], 
                           marker=profile_results["profile_marker"], linestyle='-', 
                           linewidth=3, markersize=6, 
                           color=profile_results["profile_color"], 
                           label=f"{profile_name}")
                    
                    # Plot confidence intervals if available
                    if f"{metric}_ci_lower" in profile_results and f"{metric}_ci_upper" in profile_results:
                        plt.fill_between(
                            profile_results["theta_values"],
                            profile_results[f"{metric}_ci_lower"],
                            profile_results[f"{metric}_ci_upper"],
                            color=profile_results["profile_color"],
                            alpha=0.2,
                            label=f"{profile_name} {confidence*100:.0f}% CI"
                        )
                    
                    # Find the minimum value and corresponding theta
                    values = profile_results[metric]
                    # Filter out None values before finding minimum
                    valid_values = [(theta, val) for theta, val in zip(profile_results["theta_values"], values) if val is not None]
                    
                    if valid_values:
                        thetas, vals = zip(*valid_values)
                        min_idx = np.argmin(vals)
                        min_theta = thetas[min_idx]
                        min_value = vals[min_idx]
                        min_values[profile_name] = (min_theta, min_value)
                        
                        # Add vertical line at minimum value
                        plt.axvline(x=min_theta, color="black", 
                                    linestyle='--', alpha=0.7, linewidth=2)
                        
                        # Add annotation for optimal theta value
                        plt.annotate(f'θ={min_theta:.2f}',
                                    xy=(min_theta, min_value),
                                    xytext=(10, 150),  # Offset text
                                    textcoords='offset points',
                                    fontsize=ANNOTATION_FONTSIZE,
                                    color="black",
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
                        
            plt.xlabel('Theta Value', fontsize=AXIS_LABEL_FONTSIZE)        
            plt.ylabel(metric_display_names[i], fontsize=AXIS_LABEL_FONTSIZE)
            plt.grid(True, alpha=0.3)
            plt.xticks(fontsize=TICK_FONTSIZE)
            plt.yticks(fontsize=TICK_FONTSIZE)
            
            plt.tight_layout()
            plt.show()
    
    plt.show()

"""
Function previously defined - removed duplicate.
"""

def process_single_theta_gamma(theta_idx, gamma_idx, theta, gamma, profile_params, lambda_val, num_repetitions, use_simulation, metric_names):
    """
    Helper function to process a single theta-gamma combination with all its repetitions.
    
    Args:
        theta_idx (int): Index of the theta value in the range
        gamma_idx (int): Index of the gamma value in the range
        theta (float): The theta value to process
        gamma (float): The gamma value (idle_model_timeout) to process
        profile_params (dict): Parameters for the profile
        lambda_val (float): The lambda value
        num_repetitions (int): Number of repetitions
        use_simulation (bool): Whether to use simulation or Markov model
        metric_names (list): List of metric names to record
        
    Returns:
        tuple: (theta_idx, gamma_idx, theta, gamma, results_for_all_repetitions)
    """
    
    print(f"  Processing theta = {theta:.2f}, gamma = {gamma:.4f} (theta_idx {theta_idx}, gamma_idx {gamma_idx})")
    repetition_results = {metric: [] for metric in metric_names}
    
    # Add gamma to profile params for this run
    modified_params = profile_params.copy()
    modified_params["gamma"] = gamma
    
    # Process each repetition for this theta-gamma combination
    for rep in range(num_repetitions):
        try:
            # Process this theta-gamma value for this repetition
            theta_result, metrics = process_theta(lambda_val, theta, modified_params, use_simulation)
            
            # Store the metric values for this repetition
            for metric in metric_names:
                repetition_results[metric].append(metrics[metric][0])
                
        except Exception as exc:
            print(f"    Error processing theta={theta:.2f}, gamma={gamma:.4f} (rep {rep+1}): {exc}")
            # Store None for this theta-gamma value in this repetition
            for metric in metric_names:
                repetition_results[metric].append(None)
    
    return theta_idx, gamma_idx, theta, gamma, repetition_results

def run_profiles_with_fixed_lambda(lambda_val=5.0, theta_range=None, gamma_range=None, use_simulation=False, num_repetitions=1):
    """
    Run three application profiles with a fixed lambda value and varying theta and gamma values.
    
    Args:
        lambda_val (float): The fixed arrival rate to use for all profiles
        theta_range (np.ndarray): Range of theta values to test
        gamma_range (np.ndarray): Range of gamma values (idle_model_timeout) to test
        use_simulation (bool): Whether to use simulation or Markov model
        num_repetitions (int): Number of repetitions for each theta-gamma combination
    
    Returns:
        dict: Dictionary containing results for each profile
    """
    
    num_servers = 2  # Number of servers in the system
    power_max = 150
    power_scale = 0.4
    # Define three application profiles
    profiles = [
        # {
        #     # Profile 1: CPU-intensive
        #     "name": "light-app-intensive-traffic",
        #     "color": "#1f77b4",  # blue
        #     "marker": "o",
        #     "service_rate": 5.0,
        #     "spawn_rate": 1.0,
        #     "ram_warm": 3,
        #     "cpu_warm": 1,
        #     "ram_demand": 5,
        #     "cpu_demand": 5,
        #     "num_servers": num_servers
        # },
        {
            #Profile 2: Memory-intensive application
            "name": "medium-app-medium-traffic",
            "color": "#ff7f0e",  # orange
            "marker": "s",
            "service_rate": 0.5,
            "spawn_rate": 0.5, # for 3 states: 0.25
            "spawn_model_rate": 0.5,
            "ram_warm": 10,
            "cpu_warm": 1,
            "ram_warm_model": 15,
            "cpu_warm_model": 1,
            "ram_demand": 20,
            "cpu_demand": 25,
            "num_servers": num_servers,
            "power_max": power_max,
            "power_min_scale": power_scale,
        },
        # {
        #     # Profile 3: Balanced application
        #     "name": "heavy-app-light-traffic",
        #     "color": "#2ca02c",  # green
        #     "marker": "^",
        #     "service_rate": 0.1,
        #     "spawn_rate": 0.1,
        #     "ram_warm": 30,
        #     "cpu_warm": 5,
        #     "ram_demand": 40,
        #     "cpu_demand": 50,
        #     "num_servers": num_servers
        # }
    ]
    
    # Calculate max_queue based on each profile's resource demands
    for profile in profiles:
        # Calculate how many containers can fit based on the limiting resource
        max_resource = max(profile["cpu_demand"], profile["ram_demand"])
        containers_per_server = math.floor(100 / max_resource)  # Assuming each server has 100 units of capacity
        profile["max_queue"] = containers_per_server * num_servers
        print(f"Profile '{profile['name']}': max_queue = {profile['max_queue']} (based on {max_resource} resource demand)")
    
    # Metrics to analyze
    metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
    
    # Store results for each profile
    all_results = {}
    
    # Number of CPU cores to use for multiprocessing
    num_cores = 4
    
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    for profile_idx, profile in enumerate(profiles):
        print(f"\n=== Processing Profile: {profile['name']} ===")
        
        # Extract the parameters for this profile
        profile_params = {
            "service_rate": profile["service_rate"],
            "spawn_rate": profile["spawn_rate"],
            "spawn_model_rate": profile["spawn_model_rate"],
            "max_queue": profile["max_queue"],
            "ram_warm": profile["ram_warm"],
            "cpu_warm": profile["cpu_warm"],
            "cpu_warm_model": profile["cpu_warm_model"],
            "ram_warm_model": profile["ram_warm_model"], 
            "ram_demand": profile["ram_demand"],
            "cpu_demand": profile["cpu_demand"],
            "num_servers": profile["num_servers"],
            "power_max": profile["power_max"],
            "power_min_scale": profile["power_min_scale"]
        }
        
        # Store results for this profile
        profile_results = {
            "theta_values": theta_range,
            "gamma_values": gamma_range,
            "lambda": lambda_val,
            "profile_name": profile["name"],
            "profile_color": profile["color"],
            "profile_marker": profile["marker"],
            "repetitions": num_repetitions
        }
        
        # Initialize result dictionaries for each metric
        # For 2D parameter sweep, we need a 2D grid for results
        for metric in metric_names:
            # Create a 2D array [theta][gamma] for each metric
            profile_results[metric] = np.full((len(theta_range), len(gamma_range)), None)
        
        print(f"Processing {len(theta_range)}x{len(gamma_range)} theta-gamma combinations for lambda = {lambda_val:.2f} with {num_repetitions} repetitions")
        
        # Prepare arguments for multiprocessing
        process_func = partial(
            process_single_theta_gamma,
            profile_params=profile_params,
            lambda_val=lambda_val,
            num_repetitions=num_repetitions,
            use_simulation=use_simulation,
            metric_names=metric_names
        )
        
        # Create list of arguments for each theta-gamma combination
        mp_args = []
        for theta_idx, theta in enumerate(theta_range):
            for gamma_idx, gamma in enumerate(gamma_range):
                mp_args.append((theta_idx, gamma_idx, theta, gamma))
        
        # Process theta-gamma combinations in parallel
        results = []
        with mp.Pool(processes=num_cores) as pool:
            # Use pool.starmap to process theta-gamma combinations in parallel
            results = pool.starmap(process_func, mp_args)
        
        # Organize results into the profile_results grid
        for theta_idx, gamma_idx, theta, gamma, rep_results in results:
            # For each metric, calculate the average across repetitions
            for metric in metric_names:
                # Get values for all repetitions for this theta-gamma pair
                values = rep_results[metric]
                # Calculate the mean (if there are valid values)
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    profile_results[metric][theta_idx, gamma_idx] = np.mean(valid_values)
                else:
                    profile_results[metric][theta_idx, gamma_idx] = None
        
        # Store this profile's results
        all_results[profile["name"]] = profile_results
    
    # Save results to CSV
    save_path = save_fixed_lambda_results_to_csv(all_results, profiles, lambda_val, theta_range, gamma_range)
    print(f"Results saved to {save_path}")
    
    return all_results

def save_fixed_lambda_results_to_csv(all_results, profiles, lambda_val, theta_range, gamma_range):
    """
    Save results with fixed lambda and varying theta and gamma to a CSV file.
    
    Args:
        all_results (dict): Dictionary with profile results
        profiles (list): List of profile parameters
        lambda_val (float): The lambda value used for the results
        theta_range (np.ndarray): The range of theta values
        gamma_range (np.ndarray): The range of gamma values
    
    Returns:
        str: Path to the saved CSV file
    """
    # Create a list to store all rows
    rows = []
    
    # Get metric names
    metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
    
    # Process each profile
    for profile_name, results in all_results.items():
        print(f"Saving results for profile: {profile_name}")
        
        # Find the profile parameters
        profile_params = next(p for p in profiles if p["name"] == profile_name)
        
        # For each theta and gamma value combination
        for theta_idx, theta_val in enumerate(theta_range):
            for gamma_idx, gamma_val in enumerate(gamma_range):
                row = {
                    'lambda': lambda_val,
                    'theta': theta_val,
                    'gamma': gamma_val,
                    'profile_name': profile_name,
                    'profile_color': results["profile_color"],
                    'profile_marker': results["profile_marker"],
                    'service_rate': profile_params["service_rate"],
                    'spawn_rate': profile_params["spawn_rate"],
                    'max_queue': profile_params["max_queue"],
                    'ram_warm': profile_params["ram_warm"],
                    'cpu_warm': profile_params["cpu_warm"],
                    'ram_demand': profile_params["ram_demand"],
                    'cpu_demand': profile_params["cpu_demand"]
                }
                
                # Add metric values for this theta-gamma combination
                for metric in metric_names:
                    if metric in results:
                        row[metric] = results[metric][theta_idx, gamma_idx]
                
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fixed_lambda_{lambda_val}_results_3D_{timestamp}.csv"
    
    # Ensure the directory exists
    save_dir = "./optimization_results/fixed_lambda/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + filename
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    return save_path

if __name__ == "__main__":
    print("Theta-Gamma-Lambda Visualization Tool")
    print("------------------------------------\n")
    
    print("Choose analysis mode:")
    print("1: Run fixed lambda study with theta and gamma")
    print("2: Visualize results from CSV")
    
    mode = input("Enter mode (1/2): ")
    
    if mode == "1":
        # Option 1: Run fixed lambda study
        print("\nRunning fixed lambda study...")
        lambda_val = float(input("Enter lambda value (default is 5.0): ") or "5.0")
        
        # Get theta range
        theta_points = int(input("Enter number of theta points (default is 10): ") or "10")
        theta_range = np.linspace(0.01, 1.0, theta_points)
        
        # Get gamma range
        gamma_points = int(input("Enter number of gamma points (default is 10): ") or "10")
        gamma_range = np.linspace(0.01, 1.0, gamma_points)
        # gamma_range = np.linspace(100.0, 101.0, gamma_points)
       
        # Get repetitions and simulation setting
        num_reps = int(input("Enter number of repetitions for each theta-gamma combination (default is 1): ") or "1")
        use_sim = input("Use simulation instead of Markov model? (y/n, default is n): ").lower() == "y"
        
        print(f"\nRunning with parameters:")
        print(f"- Lambda: {lambda_val}")
        print(f"- Repetitions: {num_reps}")
        print(f"- Using {'simulation' if use_sim else 'Markov model'}")
        print(f"\nTotal combinations to process: {theta_points * gamma_points}")
        
        results = run_profiles_with_fixed_lambda(
            lambda_val=lambda_val,
            theta_range=theta_range,
            gamma_range=gamma_range,
            use_simulation=use_sim,
            num_repetitions=num_reps
        )
        
        # Visualize the results
        print("\nResults processing complete.")
        plot_type = input("Choose plot type (2D/3D/both, default is both): ").lower() or "both"
        if plot_type not in ['2d', '3d', 'both']:
            plot_type = 'both'
        visualize_fixed_lambda_results(results=results, plot_type=plot_type.upper())
        
    elif mode == "2":
        # Option 2: Visualize results from CSV
        print("\nVisualizing results from CSV...")
        
        # Find CSV files in the optimization_results directory
        optimization_dir = "./optimization_results/fixed_lambda/"
        os.makedirs(optimization_dir, exist_ok=True)
        csv_files = [f for f in os.listdir(optimization_dir) if f.startswith("fixed_lambda_") and f.endswith(".csv")]
        
        if not csv_files:
            print("No fixed lambda CSV files found in the optimization_results directory.")
            exit(1)
            
        print("\nAvailable CSV files:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}: {file}")
        
        file_idx = int(input("Enter file number to visualize: ")) - 1
        if 0 <= file_idx < len(csv_files):
            csv_path = os.path.join(optimization_dir, csv_files[file_idx])
            plot_type = input("Choose plot type (2D/3D/both, default is both): ").lower() or "both"
            if plot_type not in ['2d', '3d', 'both']:
                plot_type = 'both'
            visualize_fixed_lambda_results(results_path=csv_path, plot_type=plot_type.upper())
        else:
            print("Invalid selection.")
    
    else:
        print("Invalid mode selected.")
