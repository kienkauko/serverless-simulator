import simpy
import random
import time

from System import System
from variables import *
from Topology import Topology
from Scheduler import FirstFitScheduler  # Import our schedulers

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
        yield env.timeout(total_time*2 / 100)  # Check frequently but print rarely

# --- Simulation Setup and Run ---

print("--- Simulation Start ---")
# print(f"Use Topology: {USE_TOPOLOGY}")
print(f"Simulation time is: {SIM_TIME} time units.")

# print("\n--- Cluster Configurations ---")
# for cluster_name, config in CLUSTER_CONFIG.items():
#     print(f"Cluster: {cluster_name}")
#     print(f"  Topology Node: {config['node']}")
#     print(f"  Number of Servers: {config['num_servers']}")
#     print(f"  Server CPU: {config['server_cpu']}%")
#     print(f"  Server RAM: {config['server_ram']}%")
#     print(f"  Spawn Time Factor: {config['spawn_time_factor']}")

# print("-" * 20)

random.seed(RANDOM_SEED)
env = simpy.Environment()

# Create clusters
# clusters = {}
# for cluster_name, config in CLUSTER_CONFIG.items():
    # Use the specified node for this cluster
    # node = config["node"]
    # Create the cluster with its servers
    # clusters[cluster_name] = Cluster(env, cluster_name, config, verbose=VERBOSE)

# Create topology from a file (only used if USE_TOPOLOGY is True)
topology_file = "./topology/edge.json"
cluster_file = "./topology/cluster.json"
topology = Topology(env, topology_file, cluster_file, NETWORK_MODEL)

# Select scheduler type - can be changed to BestFitScheduler for a different strategy
scheduler_class = FirstFitScheduler
# scheduler_class = BestFitScheduler  # Uncomment to use BestFitScheduler instead

# Create System with topology, clusters, and scheduler
system = System(env, topology, scheduler_class=scheduler_class)

# Call the request generator directly without wrapping in env.process()
system.request_generator(NODE_INTENSITY)

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

avg_offloaded = request_stats['offloaded_to_cloud']*100/request_stats['processed']
print(f"{'Average Offloaded to Cloud':<30}: {avg_offloaded:.2f}")
request_stats['offloaded_to_cloud']
# if request_stats['container_spawns_initiated'] > 0:
#     spawn_fail_perc = request_stats['container_spawns_failed'] / request_stats['container_spawns_initiated'] * 100
#     print(f"{'Spawn failure percentage':<30}: {spawn_fail_perc:.2f}%")

# Print average latency metrics if available
if latency_stats['count'] > 0:
    avg_total = latency_stats['total_latency'] / latency_stats['count']
    avg_prop = latency_stats['propagation_delay'] / latency_stats['count']
    avg_spawn = latency_stats['spawning_time'] / latency_stats['count']
    avg_proc  = latency_stats['processing_time'] / latency_stats['count']
    avg_wait = latency_stats['network_time'] / latency_stats['count']
    print("\n--- Average Latencies in Second---")
    print(f"{'Average Total Latency':<30}: {avg_total:.3f}")
    print(f"{'Average Propagation Delay':<30}: {avg_prop:.3f}")
    print(f"{'Average Spawn Time':<30}: {avg_spawn:.3f}")
    print(f"{'Average Processing Time':<30}: {avg_proc:.3f}")
    print(f"{'Average Network Time':<30}: {avg_wait:.3f}")

# Print application-specific statistics
# print("\n--- Application-Specific Statistics ---")
# for app_id, stats in app_stats.items():
#     print(f"\nApplication: {app_id}")
#     for key, value in stats.items():
#         print(f"  {key.replace('_', ' ').capitalize()}: {value}")
    
#     # Calculate app-specific blocking percentage
#     app_total_blocked = stats['blocked_no_server_capacity'] + stats['blocked_spawn_failed'] \
#                         +  stats['blocked_no_path']
        
#     if stats['generated'] > 0:
#         app_block_perc = app_total_blocked / stats['generated'] * 100
#         print(f"  {'Blocking percentage':<28}: {app_block_perc:.2f}%")

#     # Print app-specific latency metrics
#     if app_latency_stats[app_id]['count'] > 0:
#         app_avg_total = app_latency_stats[app_id]['total_latency'] / app_latency_stats[app_id]['count']
#         app_avg_prop = app_latency_stats[app_id]['propagation_delay'] / app_latency_stats[app_id]['count']
#         app_avg_spawn = app_latency_stats[app_id]['spawning_time'] / app_latency_stats[app_id]['count']
#         app_avg_proc = app_latency_stats[app_id]['processing_time'] / app_latency_stats[app_id]['count']
#         app_avg_wait = app_latency_stats[app_id]['waiting_time'] / app_latency_stats[app_id]['count']
#         print(f"  {'Average Total Latency':<28}: {app_avg_total:.2f}")
#         print(f"  {'Average Propagation Delay':<28}: {app_avg_prop:.2f}")
#         print(f"  {'Average Spawn Time':<28}: {app_avg_spawn:.2f}")
#         print(f"  {'Average Processing Time':<28}: {app_avg_proc:.2f}")
#         print(f"  {'Average Waiting Time':<28}: {app_avg_wait:.2f}")

# Print cluster-specific statistics
print("\n--- Cluster Statistics ---")
# change here to print which clusters
# Calculate averages across all clusters
total_clusters = len(topology.edge_clusters)
if total_clusters > 0:
    total_cpu_usage = 0
    total_ram_usage = 0
    total_cpu_reserve = 0
    total_ram_reserve = 0
    total_power_cluster = 0
    total_power_server = 0
    
    for cluster_name, cluster in topology.edge_clusters.items():
        total_cpu_usage += cluster.get_mean_cpu('cluster', 'usage')
        total_ram_usage += cluster.get_mean_ram('cluster', 'usage')
        total_cpu_reserve += cluster.get_mean_cpu('cluster', 'reserve')
        total_ram_reserve += cluster.get_mean_ram('cluster', 'reserve')
        total_power_cluster += cluster.get_mean_power('cluster')
        total_power_server += cluster.get_mean_power('server')
    
    # Calculate averages
    avg_cpu_usage = total_cpu_usage / total_clusters
    avg_ram_usage = total_ram_usage / total_clusters
    avg_cpu_reserve = total_cpu_reserve / total_clusters
    avg_ram_reserve = total_ram_reserve / total_clusters
    avg_power_cluster = total_power_cluster / total_clusters
    avg_power_server = total_power_server / total_clusters
    
    print(f"--- Average Metrics Across All Clusters ({total_clusters} clusters) ---")
    print(f"  {'Avg CPU Usage (%)':<28}: {avg_cpu_usage:.2f}%")
    print(f"  {'Avg RAM Usage (%)':<28}: {avg_ram_usage:.2f}%")
    print(f"  {'Avg CPU Reserve (%)':<28}: {avg_cpu_reserve:.2f}%")
    print(f"  {'Avg RAM Reserve (%)':<28}: {avg_ram_reserve:.2f}%")
    print(f"  {'Avg Power Usage (Cluster)':<28}: {avg_power_cluster:.2f} Watts")
    print(f"  {'Avg Power Usage (Per Server)':<28}: {avg_power_server:.2f} Watts")
else:
    print("No edge clusters available to calculate averages")

print("\n--- Topology Link Utilizations ---")
# Print link utilization data in a more readable format
if NETWORK_MODEL == "reservation":
    print("Warning: These are instantaneous values at the end of the simulation, not time-averaged.")
    link_utilization = topology.get_link_utilization()
    print("Link Utilizations (%):")
    for link, value in link_utilization.items():
        percentage = value * 100  # Convert to percentage
        print(f"  Link {link:<6}: {percentage:.2f}%")
else:
    print("Congestion path statistics:")
    for path, count in congested_paths.items():
        print(f" Congested by path {path}: {count}")
    
    # print("\nAccumulated path latency (s):")
    # for path, total_delay in accumulated_path_latency.items():
    #     print(f" Contribution of Path {path:<6}: {total_delay * 100 / latency_stats['count']:.2f}%")
    