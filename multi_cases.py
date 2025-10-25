import os
import csv
import simpy
import random
import pandas as pd
import multiprocessing
import pickle

# We need to be able to modify variables and re-run the simulation.
# We will import the modules we need to modify/reset.
import variables
import System
import Topology
import Scheduler

# --- Excel Setup ---
output_dir = './figures'
average_results_dir = os.path.join(output_dir, 'average_results')
individual_latency_dir = os.path.join(output_dir, 'individual_latency')

# Create the directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(average_results_dir, exist_ok=True)
os.makedirs(individual_latency_dir, exist_ok=True)

# Define keys for congestion results
congestion_keys = ['3-3', '3-2', '2-2', '2-1', '1-1', '1-0', '0-0']


def run_single_simulation(simulation_metrics):
    """
    Runs a single simulation with the given parameters and returns the results.
    This function re-initializes the simulation environment based on main.py logic.
    
    Args:
        simulation_metrics: Dictionary containing 'strategy', 'num_edge_server', 'traffic_intensity'
    """
    # Set the parameters from the dictionary
    variables.CLUSTER_STRATEGY = simulation_metrics['strategy']
    variables.EDGE_SERVER_NUMBER = simulation_metrics['num_edge_server']
    variables.TRAFFIC_INTENSITY = simulation_metrics['traffic_intensity']
    variables.SAVE_INDIVIDUAL_LATENCIES = simulation_metrics.get('save_individual_latencies', False)
    variables.LINK_UTILIZATION_ENABLE = simulation_metrics.get('link_utilization_enable', False)

    for key in variables.LINK_UTILIZATION:
        variables.LINK_UTILIZATION[key] = simulation_metrics['link_utilization']

    print("\n" + "="*50)
    print(f"RUNNING SIMULATION: Strategy='{variables.CLUSTER_STRATEGY}', "
          f"Servers={variables.EDGE_SERVER_NUMBER}, "
          f"Intensity={variables.TRAFFIC_INTENSITY}, "
          f"Link Utilization={simulation_metrics['link_utilization']}")
    print("="*50 + "\n")

    # --- 1. Set simulation parameters ---
    
    # Reset statistics dictionaries for a clean run by setting all values to 0
    for key in variables.request_stats:
        variables.request_stats[key] = 0
    for key in variables.latency_stats:
        variables.latency_stats[key] = 0
    for key in variables.congested_paths:
        variables.congested_paths[key] = 0
    variables.accepted_request_latencies = [] 
    
    # --- 2. Setup and Run Simulation (adapted from main.py) ---
    random.seed(variables.RANDOM_SEED)
    env = simpy.Environment()

    topology = Topology.Topology(env)
    scheduler_class = Scheduler.FirstFitScheduler
    system = System.System(env, topology, scheduler_class=scheduler_class)

    system.request_generator()
    
    env.run(until=variables.SIM_TIME)

    # --- 3. Collect Average Results (adapted from main.py) ---
    simulation_metrics.update(variables.log_ave_result())
    individual_latencies = variables.accepted_request_latencies

    # Calculate mean power, RAM, and CPU
    mean_power = 0
    mean_ram = 0
    mean_cpu = 0
    for cluster_name, cluster in topology.clusters.items():
        mean_power += cluster.get_mean_power('cluster')
        mean_ram += cluster.get_mean_ram('cluster')
        mean_cpu += cluster.get_mean_cpu('cluster')

    simulation_metrics.update({
        'mean_power': float(f"{mean_power:.1f}"),
        'mean_ram': float(f"{mean_ram:.1f}"),
        'mean_cpu': float(f"{mean_cpu:.1f}"),
    })
    
    print(f"\n--- RESULTS FOR THIS RUN ---")
    for key, value in simulation_metrics.items():
        print(f"{key}: {value}")
    
    return simulation_metrics, individual_latencies


def save_individual_latencies(sim_results, individual_latencies, individual_latency_dir):
    """
    Save individual latencies to a file if SAVE_INDIVIDUAL_LATENCIES is True.
    
    Args:
        sim_results: Dictionary containing simulation parameters and results
        individual_latencies: List of tuples (origin_node, network_delay, total_latency, bottleneck)
        individual_latency_dir: Directory to save the file
    """
    if not individual_latencies:
        return
    
    # Create filename from simulation parameters
    strategy = sim_results['strategy']
    num_server = sim_results['num_edge_server']
    intensity = sim_results['traffic_intensity']
    link_util = sim_results['link_utilization']
    
    filename = f"{strategy}_{num_server}_{intensity}_{link_util}.pkl"
    filepath = os.path.join(individual_latency_dir, filename)
    
    # Save as pickle for efficient storage and fast loading
    with open(filepath, 'wb') as f:
        pickle.dump(individual_latencies, f)
    
    print(f"Saved {len(individual_latencies)} individual latencies to {filepath}")
    
    # Also save metadata as CSV for easy inspection
    csv_filename = f"{strategy}_{num_server}_{intensity}_{link_util}_summary.csv"
    csv_filepath = os.path.join(individual_latency_dir, csv_filename)
    
    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['origin_node', 'network_delay', 'total_latency', 'bottleneck'])
        # Write only first 1000 rows to CSV for inspection (full data in pickle)
        for i, latency_data in enumerate(individual_latencies):
            if i >= 1000:  # Limit CSV to first 1000 entries
                break
            writer.writerow(latency_data)
    
    print(f"Saved summary (first 1000 entries) to {csv_filepath}")


# --- Main Loop for Multiple Cases ---
if __name__ == "__main__":

    # Iterative variables
    cases = ["centralized_cloud"]
    # intensities = [i / 100000 for i in range(10, 210, 10)] # start=0.00005, stop=0.001, step=0.0001
    intensities = [0.001, 0.004, 0.007, 0.01] # start=0.00005, stop=0.001, step=0.0001
    num_edge_servers = [5000]
    link_utilizations = [0.0]  
    # Non-iterative variables
    # --- 1. Generate all simulation tasks ---
    simulation_tasks = []
    for case in cases:
        if case.startswith("massive_edge"):
            for num_server in [5000]:
                for intensity in intensities:
                    for link_util in link_utilizations:                        
                        simulation_metrics = {
                            "strategy": case,
                            "num_edge_server": num_server,
                            "traffic_intensity": intensity,
                            "link_utilization": link_util,
                            "save_individual_latencies": True,
                            "link_utilization_enable": True
                        }
                        simulation_tasks.append((simulation_metrics,))  # Note the tuple with comma
        
        else:  # For "centralized_cloud"
            for intensity in intensities:
                for link_util in link_utilizations:
                    simulation_metrics = {
                        "strategy": case,
                        "num_edge_server": 0,
                        "traffic_intensity": intensity,
                        "link_utilization": link_util,
                        "save_individual_latencies": True,
                        "link_utilization_enable": True
                    }
                    simulation_tasks.append((simulation_metrics,))

    # --- 2. Run simulations in parallel ---
    num_processes = 4
    print(f"\nStarting {len(simulation_tasks)} simulations on {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # starmap runs the function with each tuple of arguments from the list
        results = pool.starmap(run_single_simulation, simulation_tasks)
    
    print("\nAll parallel simulations finished. Processing results...")

    # --- 3. Process results after all simulations are done ---
    main_results_list = []
    congestion_results_list = []

    for sim_results, individual_latencies in results:
        # Save individual latencies if enabled
        save_individual_latencies(sim_results, individual_latencies, individual_latency_dir)
        
        # Get the input parameters from the results
        case = sim_results['strategy']
        num_server = sim_results['num_edge_server']
        intensity = sim_results['traffic_intensity']
        link_util = sim_results['link_utilization']
        
        # Append main results
        main_results_list.append({
            'cluster_strategy': case,
            'edge_server_number': num_server,
            'traffic_intensity': intensity,
            'link_utilization': link_util,
            'blocking_percentage': sim_results['blocking_percentage'],
            'accepted_requests': sim_results['accepted_requests'],
            'avg_offloaded_to_cloud': sim_results['avg_offloaded_to_cloud'],
            'avg_total_latency': sim_results['avg_total_latency'],
            'avg_spawn_time': sim_results['avg_spawn_time'],
            'avg_processing_time': sim_results['avg_processing_time'],
            'avg_network_time': sim_results['avg_network_time'],
            'mean_power': sim_results['mean_power'],
            'mean_ram': sim_results['mean_ram'],
            'mean_cpu': sim_results['mean_cpu']
        })
    
        # Append congestion results
        congestion_row = {
            'cluster_strategy': case,
            'edge_server_number': num_server,
            'traffic_intensity': intensity,
            'link_utilization': link_util
        }
        for key in congestion_keys:
            congestion_row[key] = sim_results['congested_paths'].get(key, 0)
        congestion_results_list.append(congestion_row)

    # --- 4. Generate dynamic Excel filename based on configuration ---
    filename = f"{variables.EDGE_SERVER_PROVISION_STRATEGY}_level_{variables.EDGE_DC_LEVEL}_timeout_{variables.UNIVERSAL_TIMEOUT}.xlsx"
    excel_file_path = os.path.join(average_results_dir, filename)

    # --- 5. Save results to Excel file ---
    # Create DataFrames from the new simulation runs
    new_main_df = pd.DataFrame(main_results_list)
    new_congestion_df = pd.DataFrame(congestion_results_list)

    # Check if the file exists to append data
    if os.path.exists(excel_file_path):
        print(f"Appending results to existing file: {excel_file_path}")
        # Read the existing data
        try:
            old_main_df = pd.read_excel(excel_file_path, sheet_name='Main_Results')
            old_congestion_df = pd.read_excel(excel_file_path, sheet_name='Congestion_Results')

            # Concatenate old and new data
            main_df = pd.concat([old_main_df, new_main_df], ignore_index=True)
            congestion_df = pd.concat([old_congestion_df, new_congestion_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not read existing file {excel_file_path}. It might be corrupted. Overwriting. Error: {e}")
            main_df = new_main_df
            congestion_df = new_congestion_df
    else:
        print(f"Creating new results file: {excel_file_path}")
        main_df = new_main_df
        congestion_df = new_congestion_df


    # Write the combined (or new) data back to the Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        main_df.to_excel(writer, sheet_name='Main_Results', index=False)
        congestion_df.to_excel(writer, sheet_name='Congestion_Results', index=False)

    print(f"\nAll simulation cases are complete. Results saved to {excel_file_path}")