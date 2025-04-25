import simpy
import random
from System import System
from variables import *
from Topology import Topology
from Cluster import Cluster
from Scheduler import FirstFitScheduler, BestFitScheduler  # Import our schedulers

# --- Simulation Setup and Run ---

print("--- Simulation Start ---")
print(f"Use Topology: {USE_TOPOLOGY}")
print(f"SimTime={SIM_TIME}, Seed={RANDOM_SEED}")

print("\n--- Application Configurations ---")
for app_id, config in APPLICATIONS.items():
    print(f"App: {app_id}")
    print(f"  Arrival Rate: {config['arrival_rate']}")
    print(f"  Service Rate: {config['service_rate']}")
    print(f"  Spawn Time: {config['spawn_time']}")
    print(f"  CPU Demand: {config['min_req_cpu']}-{config['max_req_cpu']}%")
    print(f"  RAM Demand: {config['min_req_ram']}-{config['max_req_ram']}%")
    print(f"  Warm CPU: {config['min_warm_cpu']}-{config['max_warm_cpu']}%")
    print(f"  Warm RAM: {config['min_warm_ram']}-{config['max_warm_ram']}%")

print("-" * 20)

random.seed(RANDOM_SEED)
env = simpy.Environment()

# Create topology from a file (only used if USE_TOPOLOGY is True)
topology = None
cluster_node = None
if USE_TOPOLOGY:
    topology_file = "topology.csv"
    topology = Topology(topology_file)
    cluster_node = list(topology.graph.nodes)[0]
else:
    # When not using topology, we use a default placeholder node name
    cluster_node = "default_node"

# Create cluster
cluster = Cluster(env, cluster_node, NUM_SERVERS, SERVER_CPU_CAPACITY, SERVER_RAM_CAPACITY)

# Select scheduler type - can be changed to BestFitScheduler for a different strategy
scheduler_class = FirstFitScheduler
# scheduler_class = BestFitScheduler  # Uncomment to use BestFitScheduler instead

# Create System with topology, cluster, and scheduler
system = System(env, topology, cluster, use_topology=USE_TOPOLOGY,scheduler_class=scheduler_class)

# Call the request generator directly without wrapping in env.process()
system.request_generator()
env.run(until=SIM_TIME)

print("-" * 20)
print(f"--- Simulation End at time {env.now:.2f} ---")

# Print additional simulation parameters
# print("\n--- Simulation Parameters ---")
# Calculate MAX_CONTAINER (legacy calculation)
# max_containers_cpu = NUM_SERVERS * SERVER_CPU_CAPACITY / CPU_DEMAND
# max_containers_ram = NUM_SERVERS * SERVER_RAM_CAPACITY / RAM_DEMAND
# MAX_CONTAINER = min(max_containers_cpu, max_containers_ram)
# print(f"{'Max Containers':<30}: {MAX_CONTAINER:.2f}")
# print(f"{'  - By CPU':<30}: {max_containers_cpu:.2f}")
# print(f"{'  - By RAM':<30}: {max_containers_ram:.2f}")

# Calculate rates
# SPAWNING_RATE = 1/CONTAINER_SPAWN_TIME
# TIMEOUT_RATE = 1/CONTAINER_IDLE_TIMEOUT
# print(f"{'Spawning Rate':<30}: {SPAWNING_RATE:.4f}")
# print(f"{'Timeout Rate':<30}: {TIMEOUT_RATE:.4f}")
# print(f"{'Container Assign Rate':<30}: {CONTAINER_ASSIGN_RATE:.4f}")

print("\n--- Overall Simulation Statistics ---")

for key, value in request_stats.items():
    print(f"{key.replace('_', ' ').capitalize()}: {value}")

total_blocked = request_stats['blocked_no_server_capacity'] + request_stats['blocked_spawn_failed']
if USE_TOPOLOGY:
    total_blocked += request_stats['blocked_no_path']
    
total_ended = request_stats['processed'] + total_blocked
print(f"{'Total requests ended':<30}: {total_ended}")
print(f"{'Total requests counted':<30}: {latency_stats['count']}")
if request_stats['generated'] > 0:
    block_perc = total_blocked / request_stats['generated'] * 100
    print(f"{'Blocking percentage':<30}: {block_perc:.2f}%")

if request_stats['container_spawns_initiated'] > 0:
    spawn_fail_perc = request_stats['container_spawns_failed'] / request_stats['container_spawns_initiated'] * 100
    print(f"{'Spawn failure percentage':<30}: {spawn_fail_perc:.2f}%")

# Print average latency metrics if available
if latency_stats['count'] > 0:
    avg_total = latency_stats['total_latency'] / latency_stats['count']
    avg_prop = latency_stats['propagation_delay'] / latency_stats['count']
    avg_spawn = latency_stats['spawning_time'] / latency_stats['count']
    avg_proc  = latency_stats['processing_time'] / latency_stats['count']
    print("\n--- Average Latencies ---")
    print(f"{'Average Total Latency':<30}: {avg_total:.2f}")
    print(f"{'Average Propagation Delay':<30}: {avg_prop:.2f}")
    print(f"{'Average Spawn Time':<30}: {avg_spawn:.2f}")
    print(f"{'Average Processing Time':<30}: {avg_proc:.2f}")

# Print application-specific statistics
print("\n--- Application-Specific Statistics ---")
for app_id, stats in app_stats.items():
    print(f"\nApplication: {app_id}")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').capitalize()}: {value}")
    
    # Calculate app-specific blocking percentage
    app_total_blocked = stats['blocked_no_server_capacity'] + stats['blocked_spawn_failed']
    if USE_TOPOLOGY:
        app_total_blocked += stats['blocked_no_path']
        
    if stats['generated'] > 0:
        app_block_perc = app_total_blocked / stats['generated'] * 100
        print(f"  {'Blocking percentage':<28}: {app_block_perc:.2f}%")

    # Print app-specific latency metrics
    if app_latency_stats[app_id]['count'] > 0:
        app_avg_total = app_latency_stats[app_id]['total_latency'] / app_latency_stats[app_id]['count']
        app_avg_prop = app_latency_stats[app_id]['propagation_delay'] / app_latency_stats[app_id]['count']
        app_avg_spawn = app_latency_stats[app_id]['spawning_time'] / app_latency_stats[app_id]['count']
        app_avg_proc = app_latency_stats[app_id]['processing_time'] / app_latency_stats[app_id]['count']
        print(f"  {'Average Total Latency':<28}: {app_avg_total:.2f}")
        print(f"  {'Average Propagation Delay':<28}: {app_avg_prop:.2f}")
        print(f"  {'Average Spawn Time':<28}: {app_avg_spawn:.2f}")
        print(f"  {'Average Processing Time':<28}: {app_avg_proc:.2f}")

print("\n--- Final Cluster Servers ---")
for server in cluster.servers:
    print(server)
    
# Print scheduler statistics
print("\n--- Scheduler Statistics ---")
scheduler_stats = system.scheduler.get_stats()
for key, value in scheduler_stats.items():
    if isinstance(value, float):
        print(f"{key.replace('_', ' ').capitalize():<30}: {value:.4f}")
    else:
        print(f"{key.replace('_', ' ').capitalize():<30}: {value}")