import os
import csv
import simpy
import random
import pandas as pd
import multiprocessing
import filelock
import gc
import variables
import System
import Topology
import Scheduler

# Define keys for congestion results
congestion_keys = ['3-3', '3-2', '2-2', '2-1', '1-1', '1-0', '0-0']

class SimulationConfig:
    """Helper class to manage simulation configuration and file paths."""
    def __init__(self, metrics):
        self.strategy = metrics['strategy']
        self.num_server = metrics['num_edge_server']
        self.intensity = metrics['traffic_intensity']
        self.link_util = metrics['link_utilization']
        self.num_dc_per_ring = metrics.get('number_dc_per_ring', 0)
        self.save_latencies = metrics.get('save_individual_latencies', False)
        self.link_util_enable = metrics.get('link_utilization_enable', False)
        self.metrics = metrics

        # --- Excel Setup ---
        self.output_dir = './figures/yolo_image/{}/'.format(variables.COUNTRY_CODE)
        self.average_results_dir = os.path.join(self.output_dir, 'average_results')
        
        if "x_per_ring" in self.strategy:
            self.individual_latency_dir = os.path.join(self.output_dir, 'individual_latency', self.strategy, str(self.num_dc_per_ring))
        elif "edge" in self.strategy:
            self.individual_latency_dir = os.path.join(self.output_dir, str(variables.EDGE_SERVER_PROVISION_STRATEGY), str(variables.EDGE_DC_LEVEL), 'individual_latency')
        else:
            self.individual_latency_dir = os.path.join(self.output_dir, 'individual_latency')

        # Create the directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.average_results_dir, exist_ok=True)
        os.makedirs(self.individual_latency_dir, exist_ok=True)

    def apply_to_variables(self):
        """Apply configuration to the global variables module."""
        variables.CLUSTER_STRATEGY = self.strategy
        variables.EDGE_SERVER_NUMBER = self.num_server
        variables.NUM_DC_PER_RING = self.num_dc_per_ring
        variables.REQ_PER_PERSON = self.intensity
        variables.SAVE_INDIVIDUAL_LATENCIES = self.save_latencies
        variables.LINK_UTILIZATION_ENABLE = self.link_util_enable
        
        for key in variables.LINK_UTILIZATION:
            variables.LINK_UTILIZATION[key] = self.link_util

    def get_parquet_path(self):
        """Generate the path for the parquet file."""
        filename = f"{self.strategy}_{self.num_server}_{self.intensity}_{self.link_util}.parquet"
        return os.path.join(self.individual_latency_dir, filename)

    def __str__(self):
        return (f"Strategy='{self.strategy}', Servers={self.num_server}, "
                f"Intensity={self.intensity}, Link Utilization={self.link_util}")

def run_single_simulation(simulation_metrics):
    """
    Runs a single simulation with the given parameters and returns the results.
    """
    config = SimulationConfig(simulation_metrics)
    config.apply_to_variables()

    print("\n" + "="*50)
    print(f"RUNNING SIMULATION: {config}")
    print("="*50 + "\n")

    # --- 1. Reset statistics ---
    for key in variables.request_stats:
        variables.request_stats[key] = 0
    for key in variables.latency_stats:
        variables.latency_stats[key] = 0
    for key in variables.congested_paths:
        variables.congested_paths[key] = 0
    variables.accepted_request_latencies = [] 
    
    # --- 2. Setup and Run Simulation ---
    random.seed(variables.RANDOM_SEED)
    env = simpy.Environment()

    parquet_filepath = config.get_parquet_path()

    topology = Topology.Topology(env)
    scheduler_class = Scheduler.FirstFitScheduler
    # Pass the parquet file path to System so it can log requests periodically
    system = System.System(env, topology, scheduler_class=scheduler_class, trace_path=parquet_filepath)

    system.request_generator()
    
    env.run(until=variables.SIM_TIME)
    
    # Flush any remaining latencies
    if variables.SAVE_INDIVIDUAL_LATENCIES:
        system.flush_latencies()

    # --- 3. Collect Average Results ---
    simulation_metrics.update(variables.log_ave_result())
    
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
    
    # Explicitly delete large objects and force garbage collection
    del system
    del topology
    del env
    gc.collect()
    
    return simulation_metrics


# def save_individual_latencies(sim_results, individual_latencies, individual_latency_dir):
#     """
#     Save individual latencies to a file if SAVE_INDIVIDUAL_LATENCIES is True.
    
#     Args:
#         sim_results: Dictionary containing simulation parameters and results
#         individual_latencies: List of tuples (origin_node, network_delay, total_latency, bottleneck)
#         individual_latency_dir: Directory to save the file
#     """
#     if not individual_latencies:
#         return
    
#     # Create filename from simulation parameters
#     strategy = sim_results['strategy']
#     num_server = sim_results['num_edge_server']
#     intensity = sim_results['traffic_intensity']
#     link_util = sim_results['link_utilization']
    
#     filename = f"{strategy}_{num_server}_{intensity}_{link_util}.pkl"
#     filepath = os.path.join(individual_latency_dir, filename)
    
#     # Save as pickle for efficient storage and fast loading
#     with open(filepath, 'wb') as f:
#         pickle.dump(individual_latencies, f)
    
#     print(f"Saved {len(individual_latencies)} individual latencies to {filepath}")
    


def save_single_result(sim_results):
    """
    Callback function to save results immediately when a simulation finishes.
    """
    # Use the config helper to extract parameters easily
    config = SimulationConfig(sim_results)
    
    # Prepare main results row
    main_result = {
        'cluster_strategy': config.strategy,
        'edge_server_number': config.num_server,
        'number_dc_per_ring': config.num_dc_per_ring,
        'traffic_intensity': config.intensity,
        'link_utilization': config.link_util,
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
        'cluster_strategy': config.strategy,
        'edge_server_number': config.num_server,
        'traffic_intensity': config.intensity,
        'link_utilization': config.link_util
    }
    for key in congestion_keys:
        congestion_result[key] = sim_results['congested_paths'].get(key, 0)
    
    # Generate Excel filename
    filename = f"{variables.EDGE_SERVER_PROVISION_STRATEGY}_level_{variables.EDGE_DC_LEVEL}_timeout_{variables.UNIVERSAL_TIMEOUT}.xlsx"
    excel_file_path = os.path.join(config.average_results_dir, filename)
    
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
            
            print(f"✓ Saved results to {excel_file_path} (Strategy: {config.strategy}, Intensity: {config.intensity})")
    
    except filelock.Timeout:
        print(f"✗ Timeout waiting for file lock on {excel_file_path}. Skipping save for this result.")
    
    # Return a minimal summary
    return {'strategy': config.strategy, 'intensity': config.intensity, 'status': 'completed'}


# --- Main Loop for Multiple Cases ---
if __name__ == "__main__":

    # Iterative variables
    cases = ["centralized_cloud"] # Options: "massive_edge_cloud", "centralized_cloud", "x_per_ring_edge"
    # intensities = [i / 100000 for i in range(10, 210, 10)] # start=0.00005, stop=0.001, step=0.0001
    # intensities = [0.0005, 0.00055] # start=0.00005, stop=0.001, step=0.0001
    intensities = [0.0005, 0.001, 0.0015, 0.002, 0.0025] # start=0.00005, stop=0.001, step=0.0001
    # intensities = [0.0005, 0.00075, 0.001, 0.00125, 0.0015] # start=0.00005, stop=0.001, step=0.0001
    num_dc_per_ring_options = [0]  # For 'x_per_ring' strategies
    num_edge_servers = [0]  # Number of edge servers to test
    link_utilizations = [0]  
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
                        "save_individual_latencies": True,
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
    num_processes = len(intensities)
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
    # maxtasksperchild=1 forces the worker process to restart after each task, releasing all RAM to the OS
    with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
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
