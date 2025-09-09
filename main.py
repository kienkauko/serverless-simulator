import simpy
import random
from System import System
from variables import *
from Topology import Topology
from Cluster import Cluster
import time
from Scheduler import FirstFitScheduler, BestFitScheduler  # Import our schedulers

# --- Progress Tracking ---
def track_progress(env, total_time):
    """Track and print simulation progress at 10% intervals with real time measurements"""
    last_percentage = -1  # To track the last printed percentage
    start_time = time.time()
    last_real_time = start_time
    
    # Run until simulation ends
    while True:
        current_percentage = int((env.now / total_time) * 10) * 10  # Round to nearest 10%
        
        # Print only when we reach a new 10% milestone
        if current_percentage > last_percentage and current_percentage <= 100:
            current_real_time = time.time()
            elapsed_since_last = current_real_time - last_real_time
            elapsed_total = current_real_time - start_time
            
            print(f"Simulation progress: {current_percentage}% complete - Real time: {elapsed_since_last:.2f}s for this 10%, {elapsed_total:.2f}s total")
            
            last_percentage = current_percentage
            last_real_time = current_real_time
            
        # Stop when simulation is complete
        if env.now >= total_time:
            # Make sure we print 100% at the end if we haven't already
            if last_percentage < 100:
                current_real_time = time.time()
                elapsed_since_last = current_real_time - last_real_time
                elapsed_total = current_real_time - start_time
                print(f"Simulation progress: 100% complete - Real time: {elapsed_since_last:.2f}s for this 10%, {elapsed_total:.2f}s total")
            break
            
        # Check progress every small time step
        yield env.timeout(total_time / 100)  # Check frequently but print rarely

# --- Simulation Setup and Run ---

print("--- Simulation Start ---")
print(f"Use Topology: {USE_TOPOLOGY}")
print(f"SimTime={SIM_TIME}, Seed={RANDOM_SEED}")

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

# Create clusters
clusters = {}
for cluster_name, config in CLUSTER_CONFIG.items():
    # Use the specified node for this cluster
    node = config["node"]
    # Create the cluster with its servers
    clusters[cluster_name] = Cluster(env, cluster_name, config, verbose=VERBOSE)

# Create topology from a file (only used if USE_TOPOLOGY is True)
topology = None
if USE_TOPOLOGY:
    topology_file = "./topology/edge.json"
    topology = Topology(topology_file, clusters)

# Select scheduler type - can be changed to BestFitScheduler for a different strategy
scheduler_class = FirstFitScheduler
# scheduler_class = BestFitScheduler  # Uncomment to use BestFitScheduler instead

# Create System with topology, clusters, and scheduler
system = System(env, topology, clusters, use_topology=USE_TOPOLOGY, scheduler_class=scheduler_class, verbose=VERBOSE)

# Call the request generator directly without wrapping in env.process()
system.request_generator()

env.process(track_progress(env, SIM_TIME))

env.run(until=SIM_TIME)

print("-" * 20)
print(f"--- Simulation End at time {env.now:.2f} ---")
print("\n--- Overall Simulation Statistics ---")

for key, value in request_stats.items():
    print(f"{key.replace('_', ' ').capitalize()}: {value}")

total_blocked = request_stats['blocked_no_server_capacity'] + request_stats['blocked_spawn_failed'] \
                + request_stats['blocked_no_path']
    
# total_ended = request_stats['processed'] + total_blocked
# print(f"{'Total requests generated':<30}: {total_ended}")
print(f"{'Total requests counted':<30}: {latency_stats['count']}")
if request_stats['generated'] > 0:
    block_perc = total_blocked / request_stats['generated'] * 100
    print(f"{'Blocking percentage':<30}: {block_perc:.2f}%")
    if block_perc > 0:
        print(f"  {' - Due to computing':<28}: {request_stats['blocked_no_server_capacity'] / request_stats['generated'] * 100}")
        print(f"  {' - Due to link':<28}: {request_stats['blocked_no_path'] / request_stats['generated'] * 100}")

# if request_stats['container_spawns_initiated'] > 0:
#     spawn_fail_perc = request_stats['container_spawns_failed'] / request_stats['container_spawns_initiated'] * 100
#     print(f"{'Spawn failure percentage':<30}: {spawn_fail_perc:.2f}%")

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
    app_total_blocked = stats['blocked_no_server_capacity'] + stats['blocked_spawn_failed'] \
                        +  stats['blocked_no_path']
        
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
    
    # Average CPU and RAM utilization
    avg_cpu_real = cluster.get_mean_cpu('cluster', 'usage')
    avg_ram_real = cluster.get_mean_ram('cluster', 'usage')
    avg_cpu_reserve = cluster.get_mean_cpu('cluster', 'reserve')
    avg_ram_reserve = cluster.get_mean_ram('cluster', 'reserve')
    avg_power_cluster = cluster.get_mean_power('cluster')
    avg_power_server = cluster.get_mean_power('server')

    print(f"  {'Avg CPU Usage (%)':<28}: {avg_cpu_real:.2f}%")
    print(f"  {'Avg RAM Usage (%)':<28}: {avg_ram_real:.2f}%")
    print(f"  {'Avg CPU Reserve (%)':<28}: {avg_cpu_reserve:.2f}%")
    print(f"  {'Avg RAM Reserve (%)':<28}: {avg_ram_reserve:.2f}%")
    print(f"  {'Avg Power Usage (Cluster)':<28}: {avg_power_cluster:.2f} Watts")
    print(f"  {'Avg Power Usage (Per Server)':<28}: {avg_power_server:.2f} Watts")
    # Count active containers in this cluster
    # active_containers = sum(len(server.containers) for server in cluster.servers)
    # print(f"  {'Active Containers':<28}: {active_containers}")

# Add progress tracking process
