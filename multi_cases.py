import os
import csv
import simpy
import random
import pandas as pd
import multiprocessing
import pickle
import filelock

# We need to be able to modify variables and re-run the simulation.
# We will import the modules we need to modify/reset.
import variables
import System
import Topology
import Scheduler

# --- Excel Setup ---
output_dir = './figures/{}'.format(variables.COUNTRY_CODE)
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
    variables.NUM_DC_PER_RING = simulation_metrics.get('number_dc_per_ring', 0)
    variables.REQ_PER_PERSON = simulation_metrics['traffic_intensity']
    variables.SAVE_INDIVIDUAL_LATENCIES = simulation_metrics.get('save_individual_latencies', False)
    variables.LINK_UTILIZATION_ENABLE = simulation_metrics.get('link_utilization_enable', False)

    for key in variables.LINK_UTILIZATION:
        variables.LINK_UTILIZATION[key] = simulation_metrics['link_utilization']

    print("\n" + "="*50)
    print(f"RUNNING SIMULATION: Strategy='{variables.CLUSTER_STRATEGY}', "
          f"Servers={variables.EDGE_SERVER_NUMBER}, "
          f"Intensity={variables.REQ_PER_PERSON}, "
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
    energy = 0
    ram_time = 0
    cpu_time = 0
    for cluster_name, cluster in topology.clusters.items():
        energy += cluster.total_energy_usage_area
        ram_time += cluster.total_ram_usage_area
        cpu_time += cluster.total_cpu_usage_area

    simulation_metrics.update({
        'energy': float(f"{energy:.1f}"),
        'ram_time': float(f"{ram_time:.1f}"),
        'cpu_time': float(f"{cpu_time:.1f}"),
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
    


def save_single_result(result_tuple):
    """
    Callback function to save results immediately when a simulation finishes.
    This function is called by the multiprocessing pool as soon as each task completes.
    
    Args:
        result_tuple: Tuple of (sim_results, individual_latencies) from run_single_simulation
    """
    sim_results, individual_latencies = result_tuple
    
    # Save individual latencies immediately
    save_individual_latencies(sim_results, individual_latencies, individual_latency_dir)
    
    # Get the input parameters from the results
    case = sim_results['strategy']
    num_server = sim_results['num_edge_server']
    num_dc_per_ring = sim_results.get('number_dc_per_ring', 0)
    intensity = sim_results['traffic_intensity']
    link_util = sim_results['link_utilization']
    
    # Prepare main results row
    main_result = {
        'cluster_strategy': case,
        'edge_server_number': num_server,
        'number_dc_per_ring': num_dc_per_ring,
        'traffic_intensity': intensity,
        'link_utilization': link_util,
        'blocking_percentage': sim_results['blocking_percentage'],
        'accepted_requests': sim_results['accepted_requests'],
        'avg_offloaded_to_cloud': sim_results['avg_offloaded_to_cloud'],
        'avg_total_latency': sim_results['avg_total_latency'],
        'avg_spawn_time': sim_results['avg_spawn_time'],
        'avg_processing_time': sim_results['avg_processing_time'],
        'avg_network_time': sim_results['avg_network_time'],
        'energy': sim_results['energy'],
        'ram_time': sim_results['ram_time'],
        'cpu_time': sim_results['cpu_time']
    }
    
    # Prepare congestion results row
    congestion_result = {
        'cluster_strategy': case,
        'edge_server_number': num_server,
        'traffic_intensity': intensity,
        'link_utilization': link_util
    }
    for key in congestion_keys:
        congestion_result[key] = sim_results['congested_paths'].get(key, 0)
    
    # Generate Excel filename
    filename = f"{variables.EDGE_SERVER_PROVISION_STRATEGY}_level_{variables.EDGE_DC_LEVEL}_timeout_{variables.UNIVERSAL_TIMEOUT}.xlsx"
    excel_file_path = os.path.join(average_results_dir, filename)
    
    # Thread-safe file writing using a lock
    lock_file = excel_file_path + '.lock'
    lock = filelock.FileLock(lock_file, timeout=30)
    
    try:
        with lock:
            # Read existing data or create new DataFrames
            if os.path.exists(excel_file_path):
                try:
                    main_df = pd.read_excel(excel_file_path, sheet_name='Main_Results')
                    congestion_df = pd.read_excel(excel_file_path, sheet_name='Congestion_Results')
                except Exception as e:
                    print(f"Warning: Could not read {excel_file_path}. Creating new file. Error: {e}")
                    main_df = pd.DataFrame()
                    congestion_df = pd.DataFrame()
            else:
                main_df = pd.DataFrame()
                congestion_df = pd.DataFrame()
            
            # Append new results
            main_df = pd.concat([main_df, pd.DataFrame([main_result])], ignore_index=True)
            congestion_df = pd.concat([congestion_df, pd.DataFrame([congestion_result])], ignore_index=True)
            
            # Write back to Excel
            with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                main_df.to_excel(writer, sheet_name='Main_Results', index=False)
                congestion_df.to_excel(writer, sheet_name='Congestion_Results', index=False)
            
            print(f"✓ Saved results to {excel_file_path} (Strategy: {case}, Intensity: {intensity})")
    
    except filelock.Timeout:
        print(f"✗ Timeout waiting for file lock on {excel_file_path}. Skipping save for this result.")
    
    # Return a minimal summary to keep in memory (optional, for final reporting)
    return {'strategy': case, 'intensity': intensity, 'status': 'completed'}


# --- Main Loop for Multiple Cases ---
if __name__ == "__main__":

    # Iterative variables
    cases = ["centralized_cloud"] # Options: "massive_edge_cloud", "centralized_cloud"
    # intensities = [i / 100000 for i in range(10, 210, 10)] # start=0.00005, stop=0.001, step=0.0001
    intensities = [0.001, 0.002, 0.003, 0.004, 0.005] # start=0.00005, stop=0.001, step=0.0001
    # intensities = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005] # start=0.00005, stop=0.001, step=0.0001
    num_dc_per_ring_options = [0]  # For 'x_per_ring' strategies
    num_edge_servers = [15000]
    link_utilizations = [0.0]  
    # Non-iterative variables
    # --- 1. Generate all simulation tasks ---
    simulation_tasks = []
    for case in cases:
        if case == "centralized_cloud":
            for intensity in intensities:
                for link_util in link_utilizations:
                    simulation_metrics = {
                        "strategy": case,
                        "num_edge_server": 0,
                        "traffic_intensity": intensity,
                        "link_utilization": link_util,
                        "save_individual_latencies": False,
                        "link_utilization_enable": True
                    }
                    simulation_tasks.append((simulation_metrics,))

        else:
            for num_server in num_edge_servers:
                for intensity in intensities:
                    for link_util in link_utilizations:
                        for num_dc in num_dc_per_ring_options:                          
                            simulation_metrics = {
                                "strategy": case,
                                "num_edge_server": num_server,
                                "number_dc_per_ring": num_dc,
                                "traffic_intensity": intensity,
                                "link_utilization": link_util,
                                "save_individual_latencies": True,
                                "link_utilization_enable": True
                            }
                            simulation_tasks.append((simulation_metrics,))  # Note the tuple with comma
        
    # --- 2. Run simulations in parallel with immediate result saving ---
    num_processes = 5
    print(f"\nStarting {len(simulation_tasks)} simulations on {num_processes} processes...")
    print("Results will be saved immediately as each simulation completes.\n")
    
    completed_count = [0]  # Use list to allow modification in nested function
    total_count = len(simulation_tasks)
    
    def result_callback(result):
        """Callback executed when each simulation completes."""
        completed_count[0] += 1
        print(f"\n[Progress: {completed_count[0]}/{total_count} completed]")
        try:
            save_single_result(result)
        except Exception as e:
            print(f"✗ Error in result_callback while saving: {e}")
            import traceback
            traceback.print_exc()

    
    def error_callback(error):
        """Callback executed when a simulation fails."""
        print(f"\n✗ Simulation failed with error: {error}")
        completed_count[0] += 1
    
    # Use apply_async to process each result immediately as it completes
    with multiprocessing.Pool(processes=num_processes) as pool:
        async_results = []
        for task_args in simulation_tasks:
            # Submit each task and attach callbacks
            async_result = pool.apply_async(
                run_single_simulation,
                args=task_args,
                callback=result_callback,
                error_callback=error_callback
            )
            async_results.append(async_result)
        
        # Wait for all tasks to complete
        for async_result in async_results:
            async_result.wait()
    
    print("\n" + "="*60)
    print(f"All {total_count} simulations completed!")
    print("="*60)