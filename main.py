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
    print(f"  Base Spawn Time: {config['base_spawn_time']}")
    print(f"  CPU Demand: {config['min_req_cpu']}-{config['max_req_cpu']}%")
    print(f"  RAM Demand: {config['min_req_ram']}-{config['max_req_ram']}%")
    print(f"  Warm CPU: {config['min_warm_cpu']}-{config['max_warm_cpu']}%")
    print(f"  Warm RAM: {config['min_warm_ram']}-{config['max_warm_ram']}%")

print("\n--- Cluster Configurations ---")
for cluster_name, config in CLUSTER_CONFIG.items():
    print(f"Cluster: {cluster_name}")
    print(f"  Topology Node: {config['node']}")
    print(f"  Number of Servers: {config['num_servers']}")
    print(f"  Server CPU: {config['server_cpu']}%")
    print(f"  Server RAM: {config['server_ram']}%")
    print(f"  Spawn Time Factor: {config['spawn_time_factor']}")

print("-" * 20)

random.seed(RANDOM_SEED)
env = simpy.Environment()

# Create topology from a file (only used if USE_TOPOLOGY is True)
topology = None
if USE_TOPOLOGY:
    topology_file = "topology.csv"
    topology = Topology(topology_file)
# else:
#     # Create a dummy topology with just the nodes we need
#     topology = Topology("topology.csv")  # We'll still create it but might not use it for routing

# Create clusters
clusters = {}
for cluster_name, config in CLUSTER_CONFIG.items():
    # Use the specified node for this cluster
    node = config["node"]
    
    # Create the cluster with its servers
    clusters[cluster_name] = Cluster(env, node, config["num_servers"], config["server_cpu"], config["server_ram"])

# Select scheduler type - can be changed to BestFitScheduler for a different strategy
scheduler_class = FirstFitScheduler
# scheduler_class = BestFitScheduler  # Uncomment to use BestFitScheduler instead

# Create System with topology, clusters, and scheduler
system = System(env, topology, clusters, use_topology=USE_TOPOLOGY, scheduler_class=scheduler_class)

# Call the request generator directly without wrapping in env.process()
system.request_generator()
env.run(until=SIM_TIME)

print("-" * 20)
print(f"--- Simulation End at time {env.now:.2f} ---")

print("\n--- Overall Simulation Statistics ---")

for key, value in request_stats.items():
    print(f"{key.replace('_', ' ').capitalize()}: {value}")

total_blocked = request_stats['blocked_no_server_capacity'] + request_stats['blocked_spawn_failed']
if USE_TOPOLOGY:
    total_blocked += request_stats['blocked_no_path']
    
# total_ended = request_stats['processed'] + total_blocked
# print(f"{'Total requests generated':<30}: {total_ended}")
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
    avg_wait = latency_stats['waiting_time'] / latency_stats['count']
    print("\n--- Average Latencies ---")
    print(f"{'Average Total Latency':<30}: {avg_total:.2f}")
    print(f"{'Average Propagation Delay':<30}: {avg_prop:.2f}")
    print(f"{'Average Spawn Time':<30}: {avg_spawn:.2f}")
    print(f"{'Average Processing Time':<30}: {avg_proc:.2f}")
    print(f"{'Average Waiting Time':<30}: {avg_wait:.2f}")

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
        app_avg_wait = app_latency_stats[app_id]['waiting_time'] / app_latency_stats[app_id]['count']
        print(f"  {'Average Total Latency':<28}: {app_avg_total:.2f}")
        print(f"  {'Average Propagation Delay':<28}: {app_avg_prop:.2f}")
        print(f"  {'Average Spawn Time':<28}: {app_avg_spawn:.2f}")
        print(f"  {'Average Processing Time':<28}: {app_avg_proc:.2f}")
        print(f"  {'Average Waiting Time':<28}: {app_avg_wait:.2f}")

# Print cluster-specific statistics
print("\n--- Cluster Statistics ---")
for cluster_name, cluster in clusters.items():
    print(f"\nCluster: {cluster_name} at node {cluster.node}")
    print(f"  {'Number of Servers':<28}: {len(cluster.servers)}")
    
    # Count active containers in this cluster
    active_containers = sum(len(server.containers) for server in cluster.servers)
    print(f"  {'Active Containers':<28}: {active_containers}")
    
    # Print individual server status
    print(f"  {'Servers':<28}:")
    for server in cluster.servers:
        print(f"    {server}")
        for container in server.containers:
            print(f"      {container}")
    
    # Print scheduler statistics for this cluster
    print(f"  {'Scheduler Statistics':<28}:")
    scheduler_stats = system.schedulers[cluster_name].get_stats()
    for key, value in scheduler_stats.items():
        if isinstance(value, float):
            print(f"    {key.replace('_', ' ').capitalize():<26}: {value:.4f}")
        else:
            print(f"    {key.replace('_', ' ').capitalize():<26}: {value}")