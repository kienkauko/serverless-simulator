import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import copy
import os
import argparse
import glob

from variables import config as base_config
from Server import Server
from System import System
from Request import Request
from Container import Container

def run_simulation(config):
    """Run a single simulation with the given configuration and return the results."""
    # Initialize stats dictionaries
    from variables import request_stats, latency_stats
    
    # Reset statistics for each run
    for key in request_stats:
        request_stats[key] = 0
    for key in latency_stats:
        latency_stats[key] = 0.0
    
    # Create the simulation environment
    env = simpy.Environment()
    
    # Create the System with config dictionary
    system = System(env, config, distribution=config["system"]["distribution"], verbose=False)
    
    # Add servers directly to the system
    for i in range(config["system"]["num_servers"]):
        server = Server(env, f"Server-{i}", config["server"]["cpu_capacity"], config["server"]["ram_capacity"])
        system.add_server(server)
    
    # Start the request generation process
    env.process(system.request_generator())
    
    # Start the resource monitoring process
    env.process(system.resource_monitor_process())
    
    # Run the simulation
    env.run(until=config["system"]["sim_time"])
    
    # Calculate metrics
    blocking_probability = 0
    if request_stats['generated'] > 0:
        blocking_probability = request_stats['blocked_no_server_capacity'] / request_stats['generated']
    
    avg_waiting_time = 0
    if latency_stats['count'] > 0:
        avg_waiting_time = latency_stats['waiting_time'] / latency_stats['count']
    
    mean_cpu_usage = system.get_mean_cpu_usage()
    mean_ram_usage = system.get_mean_ram_usage()
    
    return {
        'blocking_probability': blocking_probability,
        'avg_waiting_time': avg_waiting_time,
        'mean_cpu_usage': mean_cpu_usage,
        'mean_ram_usage': mean_ram_usage
    }

def study_arrival_rates(test_case):
    """Run multiple simulations with varying arrival rates and analyze the results."""
    # Define the range of arrival rates to test
    arrival_rates = np.linspace(1, 30, 15)  # From 1 to 30 in 15 steps
    
    # Create a copy of the base configuration
    test_config = copy.deepcopy(base_config)

    test_config["system"]["verbose"] = False  # Disable verbose output for batch runs
    test_config["system"]["num_servers"] = 10
    test_config["system"]["sim_time"] = 1000
    test_config["system"]["distribution"] = "deterministic"  # Use deterministic for consistent results
    test_config["container"]["spawn_time"] = 4       # Time units to spawn a container
    test_config["container"]["load_model_time"] = 5      # Time units to load a model into a container
    test_config["container"]["load_request_time"] = 0.001  # Time units to load a request into a container
    # Test case 1: always on 
    if(test_case == 1):
        test_config["container"]["idle_cpu_timeout"] = 2  # Time units an idle container waits before removal
        test_config["container"]["idle_model_timeout"] = 1000  # Time units an idle model waits before removal
    # Test case 1: always off 
    elif(test_case == 2):
        test_config["container"]["idle_cpu_timeout"] = 0.001
        test_config["container"]["idle_model_timeout"] = 0.001
    # Test case 3: 3 states 
    elif(test_case == 3):
        idle_cpu_timeout = np.linspace(0.001, 10, 10)
        # test_config["container"]["idle_cpu_timeout"] = 1000
        test_config["container"]["idle_model_timeout"] = 0.001
    else:
        pass
    results = []
    
    print(f"Running simulations with varying arrival rates...")
    
    # Run simulation for each arrival rate
    if test_case == 3:
        results = []
        for idle_cpu in idle_cpu_timeout:
            print(f"Testing idle_cpu_timeout: {idle_cpu:.3f}, model_timeout: {test_config['container']['idle_model_timeout']:.3f}")
            test_config["container"]["idle_cpu_timeout"] = idle_cpu
            
            for rate in arrival_rates:
                rate = round(rate, 2)
                print(f"  Testing arrival rate: {rate:.2f}")
                test_config["request"]["arrival_rate"] = rate
                sim_results = run_simulation(test_config)
                
                # Combine both parameters with the simulation results
                result_row = {
                    'idle_cpu_timeout': idle_cpu,
                    'arrival_rate': rate
                }
                result_row.update(sim_results)
                results.append(result_row)
    else:   
        for rate in arrival_rates:
            print(f"Testing arrival rate: {rate:.2f}")
            test_config["request"]["arrival_rate"] = rate
            sim_results = run_simulation(test_config)
            
            # Combine the arrival rate with the simulation results
            result_row = {'arrival_rate': rate}
            result_row.update(sim_results)
            results.append(result_row)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV with timestamp in the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create directory if it doesn't exist
    results_dir = "multi_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save file to the directory
    filename = os.path.join(results_dir, f"test{test_case}_{timestamp}.csv")
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    return df, filename

def plot_results(df):
    """Create a dual-axis plot from the simulation results."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left y-axis (0-1 scale)
    ax1.set_xlabel('Arrival Rate (λ)')
    ax1.set_ylabel('Value (0-1 range)', color='black')
    ax1.set_ylim([-0.1, 1.1])  # Set y-axis range with a bit of margin
    
    # Plot metrics on left axis (0-1 scale)
    ax1.plot(df['arrival_rate'], df['blocking_probability'], 'o-', color='red', label='Blocking Probability')
    ax1.plot(df['arrival_rate'], df['mean_cpu_usage']/100, 's-', color='blue', label='Mean CPU Usage')
    ax1.plot(df['arrival_rate'], df['mean_ram_usage']/100, '^-', color='green', label='Mean RAM Usage')
    
    # Create right y-axis for waiting time
    ax2 = ax1.twinx()
    ax2.set_ylabel('Waiting Time (time units)', color='purple')
    ax2.plot(df['arrival_rate'], df['avg_waiting_time'], 'd-', color='purple', label='Avg Waiting Time')
    ax2.set_ylim([0, 11])  # Set y-axis range with a bit of margin

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Effect of Arrival Rate on System Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_results_test_3(df):
    """Create plots showing metrics against idle_cpu_timeout for different arrival rates.
    
    This function creates three separate figures, one for each arrival rate.
    Each figure shows blocking probability, CPU usage, RAM usage on the left y-axis (0-1 scale),
    and waiting time on the right y-axis (uncapped).
    
    Args:
        df: DataFrame containing simulation results with 'idle_cpu_timeout', 'arrival_rate',
            and the various metrics columns
    """
    # Define the three arrival rates we want to analyze
    arrival_rates = [1.0]  # Selected arrival rates to analyze
    
    # Create a figure for each arrival rate
    for rate in arrival_rates:
        # Filter data for this specific arrival rate
        rate_df = df[df['arrival_rate'] == rate]
        
        # Skip if we don't have data for this rate
        if len(rate_df) == 0:
            print(f"No data available for arrival rate {rate}")
            continue
        
        # Sort by idle_cpu_timeout to ensure proper line plotting
        rate_df = rate_df.sort_values('idle_cpu_timeout')
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Left y-axis (0-1 scale)
        ax1.set_xlabel('Idle CPU Timeout')
        ax1.set_ylabel('Value (0-1 range)', color='black')
        ax1.set_ylim([0, 1.1])  # Set y-axis range with a bit of margin
        
        # Plot metrics on left axis (0-1 scale)
        ax1.plot(rate_df['idle_cpu_timeout'], rate_df['blocking_probability'], 'o-', color='red', label='Blocking Probability')
        ax1.plot(rate_df['idle_cpu_timeout'], rate_df['mean_cpu_usage']/100, 's-', color='blue', label='Mean CPU Usage')
        ax1.plot(rate_df['idle_cpu_timeout'], rate_df['mean_ram_usage']/100, '^-', color='green', label='Mean RAM Usage')
        
        # Create right y-axis for waiting time
        ax2 = ax1.twinx()
        ax2.set_ylabel('Waiting Time (time units)', color='purple')
        ax2.plot(rate_df['idle_cpu_timeout'], rate_df['avg_waiting_time'], 'd-', color='purple', label='Avg Waiting Time')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'Effect of Idle CPU Timeout on System Performance (Arrival Rate = {rate})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.show()

def find_latest_result_file(test_case):
    """Find the latest test1_*.csv file in the current directory."""
    files = glob.glob(f"./multi_test_results/test{test_case}_*.csv")
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_from_csv(csv_path=None, test_case=4):
    """Load results from a CSV file."""
    if csv_path is None:
        csv_path = find_latest_result_file(test_case)
        if csv_path is None:
            print("No existing result file found.")
            return None
    
    try:
        print(f"Loading results from {csv_path}")
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run serverless simulator with varying arrival rates or plot existing results.')
    parser.add_argument('--plot-only', action='store_true', help='Plot results from existing CSV file without running simulations')
    parser.add_argument('--csv-file', type=str, help='Specific CSV file to load (default: latest test1_*.csv)')
    args = parser.parse_args()

    test_case = 3

    if args.plot_only:
        # Plot-only mode
        print("Plot-only mode: Loading results from CSV file...")
        results_df = load_from_csv(args.csv_file, test_case)
        
        if results_df is not None:
            print("Creating plot from loaded data...")
            if test_case == 3:
                fig = plot_results_test_3(results_df)
            else:  # idle_cpu_timeout
                fig = plot_results(results_df)
            plt.show()
        else:
            print("Failed to load results. Cannot create plot.")
    else:
        # Run the study to generate new results
        print("Running simulations to generate new results...")
        results_df, csv_file = study_arrival_rates(test_case)
        
        # Plot the results
        if test_case == 3:
            pass
                # fig = plot_results_test_3(results_df)
        else:  # idle_cpu_timeout
            fig = plot_results(results_df)
        plt.show()
        
        print("Study completed successfully!")