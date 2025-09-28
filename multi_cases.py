import os
import csv
import simpy
import time
import random
import importlib
import pandas as pd

# We need to be able to modify variables and re-run the simulation.
# We will import the modules we need to modify/reset.
import variables
import System
import Topology
import Scheduler

# --- Excel Setup ---
output_dir = './figures'
excel_file_path = os.path.join(output_dir, 'simulation_results.xlsx')

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define keys for congestion results
congestion_keys = ['3-3', '3-2', '2-2', '2-1', '1-1', '1-0', '0-0']


def run_single_simulation(cluster_strategy, edge_server_number, traffic_intensity):
    """
    Runs a single simulation with the given parameters and returns the results.
    This function re-initializes the simulation environment based on main.py logic.
    """
    print("\n" + "="*50)
    print(f"RUNNING SIMULATION: Strategy='{cluster_strategy}', Servers={edge_server_number}, Intensity={traffic_intensity}")
    print("="*50 + "\n")

    # --- 1. Set simulation parameters ---
    # By directly modifying the imported variables module
    variables.CLUSTER_STRATEGY = cluster_strategy
    variables.EDGE_SERVER_NUMBER = edge_server_number
    variables.TRAFFIC_INTENSITY = traffic_intensity
    # variables.NODE_INTENSITY = {node: node_intensity for node in variables.NODE_INTENSITY}
    
    # Reset statistics dictionaries for a clean run by setting all values to 0
    for key in variables.request_stats:
        variables.request_stats[key] = 0
    for key in variables.latency_stats:
        variables.latency_stats[key] = 0
    for key in variables.congested_paths:
        variables.congested_paths[key] = 0

    # --- 2. Setup and Run Simulation (adapted from main.py) ---
    random.seed(variables.RANDOM_SEED)
    env = simpy.Environment()

    topology = Topology.Topology(env, "./topology/edge.json", "./topology/cluster.json", variables.NETWORK_MODEL)
    scheduler_class = Scheduler.FirstFitScheduler
    system = System.System(env, topology, scheduler_class=scheduler_class)

    system.request_generator(variables.NODE_INTENSITY)
    
    env.run(until=variables.SIM_TIME)

    # --- 3. Collect Results (adapted from main.py) ---
    total_blocked = variables.request_stats['blocked_no_server_capacity'] + variables.request_stats['blocked_spawn_failed'] \
                    + variables.request_stats['blocked_no_path']

    block_perc = (total_blocked / variables.request_stats['generated'] * 100)
    
    avg_offloaded = (variables.request_stats['offloaded_to_cloud'] * 100 / variables.request_stats['processed']) 

    if variables.latency_stats['count'] > 0:
        avg_total = variables.latency_stats['total_latency'] / variables.latency_stats['count']
        avg_spawn = variables.latency_stats['spawning_time'] / variables.latency_stats['count']
        avg_proc  = variables.latency_stats['processing_time'] / variables.latency_stats['count']
        avg_wait = variables.latency_stats['network_time'] / variables.latency_stats['count']
    else:
        avg_total, avg_spawn, avg_proc, avg_wait = 0, 0, 0, 0

    # Calculate mean power
    mean_power = 0
    for cluster_name, cluster in topology.clusters.items():
        mean_power += cluster.get_mean_power('cluster')

    # Get a copy of the congested paths dictionary
    congested_paths_dict = variables.congested_paths.copy()

    results = {
        'blocking_percentage': float(f"{block_perc:.2f}"),
        'avg_offloaded_to_cloud': float(f"{avg_offloaded:.2f}"),
        'avg_total_latency': float(f"{avg_total:.3f}"),
        'avg_spawn_time': float(f"{avg_spawn:.3f}"),
        'avg_processing_time': float(f"{avg_proc:.3f}"),
        'avg_network_time': float(f"{avg_wait:.3f}"),
        'mean_power': float(f"{mean_power:.1f}"),
        'congested_paths': congested_paths_dict
    }
    
    print(f"\n--- RESULTS FOR THIS RUN ---")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    return results


# --- Main Loop for Multiple Cases ---
if __name__ == "__main__":
    cases = ["massive_edge_cloud", "massive_edge"]
    # cases = ["centralized_cloud", "massive_edge_cloud", "massive_edge"]

    # intensities = [i / 100000 for i in range(10, 150, 10)] # start=0.00005, stop=0.001, step=0.0001
    intensities = [0.001, 0.0011, 0.0012, 0.0013, 0.0014] # start=0.00005, stop=0.001, step=0.0001


    # Lists to store results for later conversion to DataFrame
    main_results_list = []
    congestion_results_list = []

    for case in cases:
        if case.startswith("massive_edge"):
            for num_server in [5000]:
                for intensity in intensities:
                    # Run simulation
                    sim_results = run_single_simulation(
                        cluster_strategy=case,
                        edge_server_number=num_server,
                        traffic_intensity=intensity
                    )
                    
                    # Append main results
                    main_results_list.append({
                        'cluster_strategy': case,
                        'edge_server_number': num_server,
                        'traffic_intensity': intensity,
                        'blocking_percentage': sim_results['blocking_percentage'],
                        'avg_offloaded_to_cloud': sim_results['avg_offloaded_to_cloud'],
                        'avg_total_latency': sim_results['avg_total_latency'],
                        'avg_spawn_time': sim_results['avg_spawn_time'],
                        'avg_processing_time': sim_results['avg_processing_time'],
                        'avg_network_time': sim_results['avg_network_time'],
                        'mean_power': sim_results['mean_power']
                    })
                
                    # Append congestion results
                    congestion_row = {
                        'cluster_strategy': case,
                        'edge_server_number': num_server,
                        'traffic_intensity': intensity
                    }
                    for key in congestion_keys:
                        congestion_row[key] = sim_results['congested_paths'].get(key, 0)
                    congestion_results_list.append(congestion_row)

        else: # For "centralized_cloud"
            # EDGE_SERVER_NUMBER is not relevant, can be set to a default like 0
            edge_server_number = 0 
            for intensity in intensities:
                # Run simulation
                sim_results = run_single_simulation(
                    cluster_strategy=case,
                    edge_server_number=edge_server_number,
                    traffic_intensity=intensity
                )

                # Append main results
                main_results_list.append({
                    'cluster_strategy': case,
                    'edge_server_number': edge_server_number,
                    'traffic_intensity': intensity,
                    'blocking_percentage': sim_results['blocking_percentage'],
                    'avg_offloaded_to_cloud': sim_results['avg_offloaded_to_cloud'],
                    'avg_total_latency': sim_results['avg_total_latency'],
                    'avg_spawn_time': sim_results['avg_spawn_time'],
                    'avg_processing_time': sim_results['avg_processing_time'],
                    'avg_network_time': sim_results['avg_network_time'],
                    'mean_power': sim_results['mean_power']
                })
            
                # Append congestion results
                congestion_row = {
                    'cluster_strategy': case,
                    'edge_server_number': edge_server_number,
                    'traffic_intensity': intensity
                }
                for key in congestion_keys:
                    congestion_row[key] = sim_results['congested_paths'].get(key, 0)
                congestion_results_list.append(congestion_row)

    # --- Save results to Excel file ---
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