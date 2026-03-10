import simpy
import random
import math
import csv
import os
from datetime import datetime
from multiprocessing import Pool

from variables import config
from Server import Server
from Request import Request
from Container import Container
from System import System

# --- Configuration for Multi-Case Experiment ---
MAX_PROCESSES = 5  # Number of parallel processes
NUM_SERVERS = 100
MEAN_SERVICE_TIMES = [1] #1, 5, 10
OFFERED_LOAD_PERCENTS = [10, 40, 70, 100] # 10, 40, 70, 100
WARM_POOL_PERCENTS = list(range(10, 101, 20))  # 10, 20, 30, ..., 100
NUM_REPETITIONS = 10

def calculate_max_containers(num_servers, cpu_demand, ram_demand):
    """Calculate max number of containers based on resource constraints."""
    resource = max(ram_demand, cpu_demand)
    max_per_server = math.floor(100 / resource)
    return max_per_server * num_servers

def run_simulation(args):
    """Run a single simulation with given args tuple and return results."""
    sim_config, case_info = args
    
    # Set random seed for this run
    random.seed(case_info['random_seed'])
    """Run a single simulation with given config and return results."""
    # Create fresh environment
    env = simpy.Environment()
    
    # Create system
    system = System(env, sim_config, distribution=sim_config["distribution"], verbose=False)
    
    # Add servers
    for i in range(sim_config["system"]["num_servers"]):
        server = Server(env, f"Server-{i}", sim_config["server"])
        system.add_server(server)
    
    # Start pre-warming
    pre_warm_done = env.process(system.pre_warm())
    
    # Start request generator after pre-warming
    def start_request_generator():
        yield pre_warm_done
        env.process(system.request_generator())
    
    env.process(start_request_generator())
    env.process(system.resource_monitor_process())
    
    # Run simulation
    env.run(until=sim_config["system"]["sim_time"])
    
    # Collect results
    blocking_percentage = 0.0
    if system.request_stats['generated'] > 0:
        blocking_percentage = system.request_stats['blocked_no_server_capacity'] / system.request_stats['generated'] * 100
    
    avg_response_time = 0.0
    if system.latency_stats['count'] > 0:
        avg_response_time = system.latency_stats['total_latency'] / system.latency_stats['count']
    
    avg_job = system.get_mean_requests_in_system()
    
    avg_cpu_usage = system.get_mean_cpu_usage()
    avg_ram_usage = system.get_mean_ram_usage()
    avg_power_usage = system.get_mean_power_usage()
    
    return {
        **case_info,
        'blocking_percentage': blocking_percentage,
        'avg_response_time': avg_response_time,
        'avg_job': avg_job,
        'avg_cpu_usage': avg_cpu_usage,
        'avg_ram_usage': avg_ram_usage,
        'avg_power_usage': avg_power_usage
    }

def main():
    # Create output directory if not exists
    output_dir = "multi_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    csv_filename = os.path.join(output_dir, f"metrics_{timestamp}.csv")
    txt_filename = os.path.join(output_dir, f"variables_{timestamp}.txt")
    
    # Calculate max containers (fixed for all cases)
    cpu_demand = config["request"]["cpu_demand"]
    ram_demand = config["request"]["ram_demand"]
    max_containers = calculate_max_containers(NUM_SERVERS, cpu_demand, ram_demand)
    
    # Write static variables to txt file
    with open(txt_filename, 'w') as f:
        f.write("=== Static Configuration ===\n\n")
        f.write(f"num_servers: {NUM_SERVERS}\n")
        f.write(f"max_containers: {max_containers}\n\n")
        f.write("--- System ---\n")
        for key, value in config["system"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Distribution ---\n")
        for key, value in config["distribution"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Server ---\n")
        for key, value in config["server"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Request ---\n")
        for key, value in config["request"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Container ---\n")
        for key, value in config["container"].items():
            f.write(f"{key}: {value}\n")
    
    print(f"Static variables saved to: {txt_filename}")
    
    # Prepare CSV file
    csv_columns = [
        'offered_load_max', 'service_time', 'offered_load', 'warm_percent', 'rep',
        'blocking_percentage', 'avg_response_time', 'avg_job',
        'avg_cpu_usage', 'avg_ram_usage', 'avg_power_usage', 'power_coe'
    ]
    
    # Build all simulation tasks
    tasks = []
    for service_time in MEAN_SERVICE_TIMES:
        max_offered_load = max_containers / service_time
        service_rate = 1.0 / service_time
        
        for offered_load_percent in OFFERED_LOAD_PERCENTS:
            arrival_rate = max_offered_load * offered_load_percent / 100
            
            for warm_percent in WARM_POOL_PERCENTS:
                warm_fraction = warm_percent / 100
                
                for rep in range(1, NUM_REPETITIONS + 1):
                    random_seed = int(rep + warm_percent + offered_load_percent + max_offered_load + service_time)
                    
                    sim_config = {
                        "system": {
                            "num_servers": NUM_SERVERS,
                            "sim_time": config["system"]["sim_time"],
                            "verbose": False,
                            "warm_percent": warm_fraction,
                        },
                        "distribution": config["distribution"].copy(),
                        "server": config["server"].copy(),
                        "request": config["request"].copy(),
                        "container": config["container"].copy(),
                    }
                    sim_config["request"]["arrival_rate_mean"] = arrival_rate
                    sim_config["request"]["service_rate"] = service_rate
                    
                    case_info = {
                        'offered_load_max': max_offered_load,
                        'service_time': service_time,
                        'offered_load': offered_load_percent,
                        'warm_percent': warm_percent,
                        'rep': rep,
                        'power_coe': config["server"]["power_scale"],
                        'random_seed': random_seed
                    }
                    
                    tasks.append((sim_config, case_info))
    
    total_cases = len(tasks)
    print(f"Total cases to run: {total_cases}")
    print(f"Running with {MAX_PROCESSES} parallel processes...")
    
    # Run simulations in parallel
    with Pool(processes=MAX_PROCESSES) as pool:
        results = pool.map(run_simulation, tasks)
    
    # Write all results to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for result in results:
            # Remove random_seed from output
            result.pop('random_seed', None)
            writer.writerow(result)
    
    print(f"\nResults saved to: {csv_filename}")
    print(f"Total cases run: {total_cases}")

if __name__ == "__main__":
    main()
