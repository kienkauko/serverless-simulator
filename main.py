import simpy
import random
import sys

from variables import *
from Server import Server
from Request import Request
from Container import Container
from System import System

# --- Main Simulation Entry Point ---

# Create the simulation environment
env = simpy.Environment()

# Initialize statistics dictionaries from variables.py
if 'request_stats' not in globals():
    request_stats = {"generated": 0, "processed": 0, "container_spawns_initiated": 0, "container_spawns_succeeded": 0, "container_spawns_failed": 0, "containers_reused": 0, "containers_removed_idle": 0, "blocked_no_server_capacity": 0, "reuse_oom_failures": 0}

if 'latency_stats' not in globals():
    latency_stats = {"total_latency": 0.0, "spawning_time": 0.0, "waiting_time": 0.0, "processing_time": 0.0, "container_wait_time": 0.0, "assignment_time": 0.0, "count": 0}

# Create the System with config dictionary
system = System(env, config, distribution=config["system"]["distribution"], verbose=config["system"]["verbose"])

# Add servers directly to the system
for i in range(config["system"]["num_servers"]):
    server = Server(env, f"Server-{i}", config["server"]["cpu_capacity"], config["server"]["ram_capacity"])
    system.add_server(server)

# Start the request generation process
env.process(system.request_generator())

# Start the resource monitoring process 
env.process(system.resource_monitor_process())

# Run the simulation
print(f"\nStarting simulation with parameters:")
print(f"- Arrival Rate (λ): {config['request']['arrival_rate']} req/s")
print(f"- Service Rate (μ): {config['request']['service_rate']} req/s")
print(f"- Number of Servers: {config['system']['num_servers']}")
# print(f"- CPU Demand: {CPU_DEMAND}") # Using the backward compatibility variable for display
# print(f"- RAM Demand: {RAM_DEMAND}") # Using the backward compatibility variable for display
print(f"- Simulation Time: {config['system']['sim_time']}s")
print(f"- Container Spawn Time: {config['container']['spawn_time']}s")
print(f"- Container Idle Timeout: {config['container']['idle_cpu_timeout']}s\n")
print(f"- Model Idle Timeout: {config['container']['idle_model_timeout']}s\n")

env.run(until=config["system"]["sim_time"])

# Print final statistics
print("\n--- Simulation Statistics ---")
print(f"Requests Generated: {request_stats['generated']}")
print(f"Requests Processed: {request_stats['processed']}")
print(f"Requests Blocked (No Capacity): {request_stats['blocked_no_server_capacity']}")

# Calculate and print blocking probability
if request_stats['generated'] > 0:
    blocking_probability = request_stats['blocked_no_server_capacity'] / request_stats['generated']
    print(f"Blocking Probability: {blocking_probability:.4f} ({blocking_probability*100:.2f}%)")

print(f"Container Spawns Initiated: {request_stats['container_spawns_initiated']}")
print(f"Container Spawns Succeeded: {request_stats['container_spawns_succeeded']}")
print(f"Container Spawns Failed: {request_stats['container_spawns_failed']}")
print(f"Containers Reused: {request_stats['containers_reused']}")
print(f"Container OOM Reuse Failures: {request_stats['reuse_oom_failures']}")
print(f"Containers Removed Idle: {request_stats['containers_removed_idle']}")

# Print latency statistics
if latency_stats['count'] > 0:
    avg_total = latency_stats['total_latency'] / latency_stats['count']
    avg_spawn = latency_stats['spawning_time'] / latency_stats['count']
    avg_wait = latency_stats['waiting_time'] / latency_stats['count']
    avg_proc = latency_stats['processing_time'] / latency_stats['count']
    avg_container_wait = latency_stats['container_wait_time'] / latency_stats['count']
    avg_assignment = latency_stats['assignment_time'] / latency_stats['count']
    
    print("\n--- Average Latencies ---")
    print(f"Total End-to-End Latency: {avg_total:.2f}s")
    print(f"Container Spawn Time: {avg_spawn:.2f}s")
    print(f"Total Wait Time: {avg_wait:.2f}s")
    print(f"  - Container Wait: {avg_container_wait:.2f}s")
    print(f"  - Assignment Time: {avg_assignment:.2f}s")
    print(f"Processing Time: {avg_proc:.2f}s")

# Calculate Little's Law metrics
if system.env.now > 0:
    avg_waiting_count = system.total_waiting_area / system.env.now
    avg_processing_count = system.get_mean_processing_count()
    avg_system_count = system.get_mean_requests_in_system()
    effective_arrival_rate = request_stats['processed'] / system.env.now
    little_law_wait_time = avg_waiting_count / effective_arrival_rate if effective_arrival_rate > 0 else 0
    
    print("\n--- Queue Metrics ---")
    print(f"Average number of waiting requests: {avg_waiting_count:.2f}")
    print(f"Average number of processing requests: {avg_processing_count:.2f}")
    print(f"Average number of requests in system: {avg_system_count:.2f}")
    print(f"Effective arrival rate: {effective_arrival_rate:.2f} req/s")
    print(f"Little's Law predicted waiting time: {little_law_wait_time:.2f}s")
    print(f"Actual average waiting time from measurements: {avg_wait:.2f}s")

# Print resource utilization statistics
mean_cpu_usage = system.get_mean_cpu_usage()
mean_ram_usage = system.get_mean_ram_usage()
print("\n--- Resource Utilization ---")
print(f"Mean CPU Usage: {mean_cpu_usage:.4f}%")
print(f"Mean RAM Usage: {mean_ram_usage:.4f}%")