#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization tool for theta-lambda results
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import multiprocessing as mp
from functools import partial

from System import System
from Server import Server

import simpy
from model_3D import MarkovModel


def visualize_fixed_lambda_results(results_path=None, results=None, confidence=0.95):
    """
    Create visualizations for the fixed lambda results, showing how metrics
    vary with theta for each profile.
    
    Args:
        results_path (str): Path to the CSV file with results
        results (dict): Results dictionary (alternative to loading from CSV)
        confidence (float): Confidence level for interval calculation (default: 0.95)
    
    Returns:
        None
    """
    # Define font sizes for consistent styling
    TITLE_FONTSIZE = 16
    AXIS_LABEL_FONTSIZE = 22
    TICK_FONTSIZE = 22
    LEGEND_FONTSIZE = 18
    SUPTITLE_FONTSIZE = 18
    ANNOTATION_FONTSIZE = 18
    
    # Import scipy for confidence interval calculation
    from scipy import stats
    import numpy as np
    
    # Load results from CSV if path is provided
    if results_path and results is None:
        df = pd.read_csv(results_path,  encoding='utf-8')
        
        # Get unique profiles
        profiles = []
        for profile_name in df['profile_name'].unique():
            profile_data = df[df['profile_name'] == profile_name].iloc[0]
            profiles.append({
                "name": profile_name,
                "color": profile_data['profile_color'],
                "marker": profile_data['profile_marker'],
                "line_style": profile_data['profile_line']
            })
        
        # Reconstruct results dictionary
        results = {}
        for profile in profiles:
            profile_df = df[df['profile_name'] == profile['name']]
            
            # Check if repetition column exists
            has_repetitions = 'repetition' in profile_df.columns
            
            # Get unique theta values
            theta_values = sorted(profile_df['theta'].unique())
            
            profile_results = {
                "theta_values": theta_values,
                "lambda": profile_df['lambda'].iloc[0],
                "profile_name": profile['name'],
                "profile_color": profile['color'],
                "profile_marker": profile['marker'],
                "profile_line": profile['line_style'],
            }
            
            # Add metrics
            metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
            
            for metric in metric_names:
                if metric in profile_df.columns:
                    if has_repetitions:
                        # Store both mean and confidence intervals for each theta
                        avg_values = []
                        ci_lower = []
                        ci_upper = []
                        
                        for theta in theta_values:
                            theta_df = profile_df[profile_df['theta'] == theta]
                            metric_values = theta_df[metric].dropna().values
                            
                            if len(metric_values) > 0:
                                mean_val = np.mean(metric_values)
                                avg_values.append(mean_val)
                                
                                if len(metric_values) > 1:
                                    # Calculate confidence intervals using t-distribution
                                    t_val = stats.t.ppf((1 + confidence) / 2, len(metric_values) - 1)
                                    std_err = stats.sem(metric_values)
                                    margin = t_val * std_err
                                    ci_lower.append(mean_val - margin)
                                    ci_upper.append(mean_val + margin)
                                else:
                                    # If only one repetition, can't calculate CI
                                    ci_lower.append(mean_val)
                                    ci_upper.append(mean_val)
                            else:
                                avg_values.append(None)
                                ci_lower.append(None)
                                ci_upper.append(None)
                        
                        profile_results[metric] = avg_values
                        profile_results[f"{metric}_ci_lower"] = ci_lower
                        profile_results[f"{metric}_ci_upper"] = ci_upper
                    else:
                        profile_results[metric] = profile_df[metric].values
            
            results[profile['name']] = profile_results
    
    if not results:
        print("No results provided or loaded.")
        return
    
    # If results came directly from the run_profiles_with_fixed_lambda function with repetitions
    for profile_name, profile_results in results.items():
        for metric in ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]:
            if metric in profile_results and isinstance(profile_results[metric], list) and isinstance(profile_results[metric][0], list):
                # Calculate average metric value for each theta across repetitions
                avg_values = []
                ci_lower = []
                ci_upper = []
                
                for i in range(len(profile_results["theta_values"])):
                    values = [rep[i] for rep in profile_results[metric] if i < len(rep) and rep[i] is not None]
                    
                    if values:
                        mean_val = np.mean(values)
                        avg_values.append(mean_val)
                        
                        if len(values) > 1:
                            # Calculate confidence intervals using t-distribution
                            t_val = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
                            std_err = stats.sem(values)
                            margin = t_val * std_err
                            ci_lower.append(mean_val - margin)
                            ci_upper.append(mean_val + margin)
                        else:
                            # If only one repetition, can't calculate CI
                            ci_lower.append(mean_val)
                            ci_upper.append(mean_val)
                    else:
                        avg_values.append(None)
                        ci_lower.append(None)
                        ci_upper.append(None)
                
                profile_results[metric] = avg_values
                profile_results[f"{metric}_ci_lower"] = ci_lower
                profile_results[f"{metric}_ci_upper"] = ci_upper
    
    # Metric information for plotting
    metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
    metric_display_names = ["Blocking Probability", "Latency (s)", "RAM Per Request (%)", "CPU Per Request (%)", " Energy Per Request (J)"]
    
    # Create individual plots for each metric
    for i, metric in enumerate(metric_names):
        plt.figure(figsize=(5.5, 4.8))
        
        # Dictionary to store min values and corresponding thetas for each profile
        min_values = {}
        
        # Plot each profile
        for profile_name, profile_results in results.items():
            if metric in profile_results:
                # Find the minimum value and corresponding theta
                values = profile_results[metric]
                # Filter out None values before finding minimum
                valid_values = [(theta, val) for theta, val in zip(profile_results["theta_values"], values) if val is not None]
                
                min_theta = None
                if valid_values:
                    thetas, vals = zip(*valid_values)
                    min_idx = np.argmin(vals)
                    min_theta = thetas[min_idx]
                    min_value = vals[min_idx]
                    min_values[profile_name] = (min_theta, min_value)
                    
                    # Add vertical line at minimum value
                    # plt.axvline(x=min_theta, color="black", 
                    #             linestyle='--', alpha=0.7, linewidth=2)
                
                # Create custom legend label with best theta value
                if min_theta is not None:
                    legend_label = f"{profile_name}, best θ = {min_theta:.2f}"
                else:
                    legend_label = profile_name
                
                # Plot mean line
                line, = plt.plot(profile_results["theta_values"], profile_results[metric], 
                       linestyle=profile_results["profile_line"], 
                       linewidth=5, markersize=10, 
                       color="green", 
                       label=legend_label)
                
                # Plot confidence intervals if available
                if f"{metric}_ci_lower" in profile_results and f"{metric}_ci_upper" in profile_results:
                    plt.fill_between(
                        profile_results["theta_values"],
                        profile_results[f"{metric}_ci_lower"],
                        profile_results[f"{metric}_ci_upper"],
                        color=profile_results["profile_color"],
                        alpha=0.2,
                        # label=f"{confidence*100:.0f}% CI"
                    )
                    
        plt.xlabel('$\\theta_1$ Value', fontsize=AXIS_LABEL_FONTSIZE)        
        plt.ylabel(metric_display_names[i], fontsize=AXIS_LABEL_FONTSIZE)
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        
        plt.tight_layout()
        plt.show()
    
    plt.show()

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
            "load_request_time": 1/10000,
            "load_model_time": 2, 
            "idle_model_timeout": 0.00001,
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


def process_single_theta(theta_idx, theta, profile_params, lambda_val, num_repetitions, use_simulation, metric_names):
    """
    Helper function to process a single theta value with all its repetitions.
    
    Args:
        theta_idx (int): Index of the theta value in the range
        theta (float): The theta value to process
        profile_params (dict): Parameters for the profile
        lambda_val (float): The lambda value
        num_repetitions (int): Number of repetitions
        use_simulation (bool): Whether to use simulation or Markov model
        metric_names (list): List of metric names to record
        
    Returns:
        tuple: (theta_idx, theta, results_for_all_repetitions)
    """    
    print(f"  Processing theta = {theta:.2f} (index {theta_idx})")
    repetition_results = {metric: [] for metric in metric_names}
    
    # Process each repetition for this theta
    for rep in range(num_repetitions):
        try:
            # Process this theta value for this repetition
            theta_result, metrics = process_theta(lambda_val, theta, profile_params, use_simulation)
            
            # Store the metric values for this repetition
            for metric in metric_names:
                repetition_results[metric].append(metrics[metric][0])
                
        except Exception as exc:
            print(f"    Error processing theta={theta:.2f} (rep {rep+1}): {exc}")
            # Store None for this theta value in this repetition
            for metric in metric_names:
                repetition_results[metric].append(None)
    
    return theta_idx, theta, repetition_results

def run_profiles_with_fixed_lambda(lambda_val=5.0, theta_range=None, use_simulation=False, num_repetitions=1):
    """
    Run three application profiles with a fixed lambda value and varying theta values.
    
    Args:
        lambda_val (float): The fixed arrival rate to use for all profiles
        theta_range (np.ndarray): Range of theta values to test
        use_simulation (bool): Whether to use simulation or Markov model
        num_repetitions (int): Number of repetitions for each theta evaluation
    
    Returns:
        dict: Dictionary containing results for each profile
    """
    import math
    
    if theta_range is None:
        theta_range = np.linspace(0.01, 5.0, 100)
    
    num_servers = 10  # Number of servers in the system
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
            "color": "#0eff7a",  # orange
            "marker": "x",
            "service_rate": 0.25, # default 0.5, 0.25 is for model 3D
            "spawn_rate": 0.5,
            "ram_warm": 10,
            "cpu_warm": 1,
            "ram_demand": 17.5, # 20 for true resource
            "cpu_demand": 13, # 25 for true resource, pls change the queue size also!!!
            "ram_warm_model": 15,
            "cpu_warm_model": 1,
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
        #     "spawn_rate": 0.2,
        #     "spawn_model_rate": 0.1,
        #     "ram_warm": 10,
        #     "cpu_warm": 2,
        #     "ram_warm_model": 35,
        #     "cpu_warm_model": 2,
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
        profile["max_queue"] = 40

        print(f"Profile '{profile['name']}': max_queue = {profile['max_queue']} (based on {max_resource} resource demand)")
    
    # Metrics to analyze
    metric_names = ["blocking_ratios", "latency", "ram_usage_per_request", "cpu_usage_per_request", "power_usage_per_request"]
    
    # Store results for each profile
    all_results = {}
    
    # Determine number of CPU cores to use for multiprocessing
    # num_cores = mp.cpu_count() - 1  # Leave one core free
    num_cores = 5
    if num_cores < 1:
        num_cores = 1
    
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    for profile_idx, profile in enumerate(profiles):
        print(f"\n=== Processing Profile: {profile['name']} ===")
        
        # Extract the parameters for this profile
        profile_params = {
            "service_rate": profile["service_rate"],
            "spawn_rate": profile["spawn_rate"],
            "max_queue": profile["max_queue"],
            "ram_warm": profile["ram_warm"],
            "cpu_warm": profile["cpu_warm"],
            "ram_warm_model": profile["ram_warm_model"],
            "cpu_warm_model": profile["cpu_warm_model"],
            "ram_demand": profile["ram_demand"],
            "cpu_demand": profile["cpu_demand"],
            "num_servers": profile["num_servers"],
            "power_max": profile["power_max"],
            "power_min_scale": profile["power_min_scale"]
        }
        
        # Store results for this profile
        profile_results = {
            "theta_values": theta_range,
            "lambda": lambda_val,
            "profile_name": profile["name"],
            "profile_color": profile["color"],
            "profile_marker": profile["marker"],
            "repetitions": num_repetitions
        }
        
        # Initialize result dictionaries for each metric and repetition
        for metric in metric_names:
            profile_results[metric] = [[] for _ in range(num_repetitions)]
        
        print(f"Processing {len(theta_range)} theta values for lambda = {lambda_val:.2f} with {num_repetitions} repetitions")
        
        # Prepare arguments for multiprocessing
        process_func = partial(
            process_single_theta,
            profile_params=profile_params,
            lambda_val=lambda_val,
            num_repetitions=num_repetitions,
            use_simulation=use_simulation,
            metric_names=metric_names
        )
        
        # Create list of arguments for each theta value
        mp_args = [(i, theta) for i, theta in enumerate(theta_range)]
        
        # Process theta values in parallel
        results = []
        with mp.Pool(processes=num_cores) as pool:
            # Use pool.starmap to process theta values in parallel
            results = pool.starmap(process_func, mp_args)
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Reorganize results into profile_results format
        for _, _, rep_results in results:
            # For each metric
            for metric in metric_names:
                # For each repetition
                for rep in range(num_repetitions):
                    if rep < len(rep_results[metric]):
                        profile_results[metric][rep].append(rep_results[metric][rep])
                    else:
                        profile_results[metric][rep].append(None)
        
        # Store this profile's results
        all_results[profile["name"]] = profile_results
    
    # Save results to CSV
    save_path = save_fixed_lambda_results_to_csv(all_results, profiles, lambda_val)
    print(f"Results saved to {save_path}")
    
    return all_results

def save_fixed_lambda_results_to_csv(all_results, profiles, lambda_val):
    """
    Save results with fixed lambda and varying theta to a CSV file.
    
    Args:
        all_results (dict): Dictionary with profile results
        profiles (list): List of profile parameters
        lambda_val (float): The lambda value used for the results
    
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
        
        # Get the number of repetitions
        num_repetitions = results.get("repetitions", 1)
        
        # For each theta value
        for i, theta_val in enumerate(results["theta_values"]):
            # For each repetition
            for rep in range(num_repetitions):
                row = {
                    'lambda': lambda_val,
                    'theta': theta_val,
                    'repetition': rep + 1,
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
                
                # Add metric values for this repetition
                for metric in metric_names:
                    if metric in results and rep < len(results[metric]) and i < len(results[metric][rep]):
                        row[metric] = results[metric][rep][i]
                
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fixed_lambda_{lambda_val}_results_{timestamp}.csv"
    
    # Ensure the directory exists
    save_dir = "./optimization_results/fixed_lambda/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + filename
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    return save_path

if __name__ == "__main__":
    print("Theta-Lambda Visualization Tool")
    print("-------------------------------\n")
    
    print("Choose analysis mode:")
    print("1: Run fixed lambda study")
    print("2: Visualize results from CSV")
    
    mode = input("Enter mode (1/2): ")
    
    if mode == "1":
        # Option 1: Run fixed lambda study
        print("\nRunning fixed lambda study...")
        lambda_val = float(input("Enter lambda value (default is 5.0): ") or "5.0")
        num_points = int(input("Enter number of theta points (default is 100): ") or "100")
        num_reps = int(input("Enter number of repetitions for each theta (default is 1): ") or "1")
        use_sim = input("Use simulation instead of Markov model? (y/n, default is n): ").lower() == "y"
        
        # Create theta range with linear spacing from 0.01 to 1.0,
        # and additional specific values
        theta_range = np.concatenate([
            np.linspace(0.01, 10.0, num_points),
            # np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        ])
        results = run_profiles_with_fixed_lambda(
            lambda_val=lambda_val,
            theta_range=theta_range,
            use_simulation=use_sim,
            num_repetitions=num_reps
        )
        
        # Visualize the results
        confidence = float(input("Enter confidence level for intervals (0-1, default is 0.95): ") or "0.95")
        visualize_fixed_lambda_results(results=results, confidence=confidence)
        
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
            confidence = float(input("Enter confidence level for intervals (0-1, default is 0.95): ") or "0.95")
            visualize_fixed_lambda_results(results_path=csv_path, confidence=confidence)
        else:
            print("Invalid selection.")
    
    else:
        print("Invalid mode selected.")
