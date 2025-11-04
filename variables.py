import random
# --- Configuration ---
RANDOM_SEED = 42
SIM_TIME = 400  # Simulation time units (e.g., seconds)
METRIC_RECORDED_TIME = 0  # Time after which metrics are recorded
VERBOSE = False  # Set to True to enable detailed logging

# Container Parameters
CONTAINER_ASSIGN_RATE = 1000000.0 # Average rate for request assignment (very fast)

# Topology configuration
USE_TOPOLOGY = True  # Enable topology routing
COUNTRY_CODE = "VNM"  # Country code for topology selection
TOPOLOGY_PATH = f"./topology/countries/{COUNTRY_CODE}/result.json"
# TOPOLOGY_PATH = "./topology/Germany.json"
# CLUSTER_PATH = "./topology/cluster.json"
NETWORK_MODEL = "ps" # Options: "ps", "reservation"

CLUSTER_STRATEGY = "massive_edge"  # Options: "massive_edge", "massive_edge_cloud", "centralized_cloud", "one_per_ring_edge_cloud", "one_per_ring_edge"
CENTRAL_CLOUD = "central_cloud"  # Central cloud name in the topology
# CENTRAL_CLOUD_NODE = "12876"  # Central cloud node ID in the topology
EDGE_SERVER_NUMBER = 10000  # CPU capacity for all MECs
EDGE_DC_LEVEL = 1
EDGE_SERVER_PROVISION_STRATEGY = "equally"  # Options: "equally", "population_weighted"
CLOUD_SPAWN_TIME_FACTOR = 0.5  # Cloud spawn time multiplier (faster)
CLOUD_PROCESSING_TIME_FACTOR = 0.6  # Cloud processing time multiplier (faster)

# Define Ingress nodes for custom ingress defined in define_ingress_nodes() in Topology.py
INGRESS_MODE = 'country'  # Options: 'country', 'cities'
EXAMINED_CITIES = {
    'Frankfurt': '12860',
    # 'Munich': 12838
}

# Define utilization for connected links
BW_POPULATION_SCALE_ENABLE = False
LINK_UTILIZATION_ENABLE = True  # Enable link utilization adjustments
LINK_UTILIZATION = {
    '3-3': 0.5,
    '3-2': 0.5,
    '2-3': 0.5,
    '2-2': 0.5,
    '2-1': 0.5,
    '1-2': 0.5,
    '1-1': 0.5,
    '1-0': 0.5,
    '0-1': 0.5,
    '0-0': 0.5,
} 






# --- Multi-Cluster Configuration ---

# Traffic intensity factor to scale arrival rates based on node population
TRAFFIC_INTENSITY = 0.0005  # Adjust this factor to scale overall traffic, default: 0.0001
NODE_INTENSITY = 10  # Percentage of level 3 nodes generating traffic (0-100)
# Application definitions for heterogeneous workloads
APPLICATIONS = {
    "app1": { # a Tiktok video sent to Cloud for processing (10s)
        # "arrival_rate": 50.0,  # Requests per time unit
        "service_rate": 1/5.0,  # To compute processing time only, process may take 5s
        "base_spawn_time": 5.0,  # Base time units to spawn a container (modified by cluster factor)
        "min_warm_cpu": 0.5,  # Minimum CPU for warm container
        "max_warm_cpu": 0.5,  # Maximum CPU for warm container
        "min_warm_ram": 5.0,  # Minimum RAM for warm container
        "max_warm_ram": 5.0, # Maximum RAM for warm container
        "min_req_cpu": 20.0,  # Minimum CPU demand for request
        "max_req_cpu": 20.0,  # Maximum CPU demand for request
        "min_req_ram": 7.0,  # Minimum RAM demand for request
        "max_req_ram": 7.0,  # Maximum RAM demand for request
        # "bandwidth_direct": 40000000,  # Bandwidth demand for this application: bit per second
        # "bandwidth_indirect": 1000000,  # Bandwidth demand for this application: bit per second
        "data_location": "12876",  # Location of data for this application - Cloud node
        "packet_size_direct_upload": 163886080,  # in bits, default to 10 MB
        "packet_size_direct_download": 81920,  # in bits, default to 10 KB
        "data_path_required": False,  # Whether data path is required
        "packet_size_indirect_upload": 0,  # in bits, default to 10 MB
        "packet_size_indirect_download": 0,  # in bits, default to 10 MB
    },
    # "app2": {
    #     "arrival_rate": 30.0,
    #     "service_rate": 1.5,
    #     "base_spawn_time": 7.0,
    #     "min_warm_cpu": 1.0,
    #     "max_warm_cpu": 1.0,
    #     "min_warm_ram": 8.0,
    #     "max_warm_ram": 8.0,
    #     "min_req_cpu": 40.0,
    #     "max_req_cpu": 40.0,
    #     "min_req_ram": 40.0,
    #     "max_req_ram": 40.0,
    #     "bandwidth_direct": 10.0,  # Bandwidth demand for this application
    #     "bandwidth_indirect": 1.0,  # Bandwidth demand for this application
    #     "data_location": "nodeB",  # Location of data for this application


    # },
    # "app3": {
    #     "arrival_rate": 5.0,
    #     "service_rate": 1.0,
    #     "base_spawn_time": 10.0,
    #     "min_warm_cpu": 2.0,
    #     "max_warm_cpu": 5.0,
    #     "min_warm_ram": 10.0,
    #     "max_warm_ram": 20.0,
    #     "min_req_cpu": 60.0,
    #     "max_req_cpu": 70.0,
    #     "min_req_ram": 60.0,
    #     "max_req_ram": 70.0,
    #     "bandwidth_direct": 15.0,  # Bandwidth demand for this application
    # }
}

UNIVERSAL_TIMEOUT = 2  # Time to live for idle function - warm time

# Statistics
request_stats = {
    'generated': 0,
    'processed': 0,
    'blocked_no_server_capacity': 0, # Blocked because no server could *ever* fit it
    'blocked_spawn_failed': 0,      # Blocked because spawning failed (transient lack of resources)
    'blocked_no_path': 0,  # New: rejected due to no routing path with available bandwidth
    'offloaded_to_cloud': 0,  # New: offloaded to cloud due to lack of edge resources
    'container_spawns_initiated': 0,
    'container_spawns_failed': 0,
    'container_spawns_succeeded': 0,
    'containers_reused': 0,
    'containers_removed_idle': 0,
    'reuse_oom_failures': 0 # Out Of Memory/CPU when trying to activate reused container
    # 'blocked_no_path_level_3-3': 0, # No path between level 3 nodes
    # 'blocked_no_path_level_3-2': 0, # No path between level 3 and level 2 nodes
    # 'blocked_no_path_level_2-2': 0, # No path between level 2 nodes
    # 'blocked_no_path_level_2-1': 0, # No path between level 2 and level 1 nodes
    # 'blocked_no_path_level_1-1': 0, # No path between level 1 nodes
    # 'blocked_no_path_level_1-0': 0, # No path between level 1 and level 0 nodes
    # 'blocked_no_path_level_0-0': 0, # No path between level 0 nodes
}

congested_paths = {
    '3-3': 0,
    '3-2': 0,
    '2-2': 0,
    '2-1': 0,
    '1-1': 0,
    '1-0': 0,
    '0-0': 0,
}

accumulated_path_latency = {
    '3-3': 0,
    '3-2': 0,
    '2-2': 0,
    '2-1': 0,
    '1-1': 0,
    '1-0': 0,
    '0-0': 0,
}

SAVE_INDIVIDUAL_LATENCIES = False  # Set to True to save individual request latencies
accepted_request_latencies = []  # Will store tuples of (ingress_id, latency)

# App-specific statistics
app_stats = {}
for app_id in APPLICATIONS:
    app_stats[app_id] = {
        'generated': 0,
        'processed': 0,
        'blocked_no_server_capacity': 0,
        'blocked_spawn_failed': 0,
        'blocked_no_path': 0,
        'container_spawns_initiated': 0,
        'container_spawns_failed': 0,
        'container_spawns_succeeded': 0,
        'containers_reused': 0,
        'containers_removed_idle': 0,
        'reuse_oom_failures': 0,
        'bocked_no_path_level_3-3': 0, # No path between level 3 nodes
        'bocked_no_path_level_3-2': 0, # No path between level 3 and level 2 nodes
        'bocked_no_path_level_2-2': 0, # No path between level 2 nodes
        'bocked_no_path_level_2-1': 0, # No path between level 2 and level 1 nodes
        'bocked_no_path_level_1-1': 0, # No path between level 1 nodes
        'bocked_no_path_level_1-0': 0, # No path between level 1 and level 0 nodes
        'bocked_no_path_level_0-0': 0, # No path between level 0 nodes
    }

# Cluster-specific statistics
# cluster_stats = {}
# for cluster_name in CLUSTER_CONFIG:
#     cluster_stats[cluster_name] = {
#         'cpu_real': [],
#         'ram_real': [],
#         'cpu_reserve': [],
#         'ram_reserve': [],
#     }

# New dictionary to track latency metrics (in time units)
latency_stats = {
    'total_latency': 0.0,
    'propagation_delay': 0.0,
    'spawning_time': 0.0,
    'processing_time': 0.0,
    'waiting_time': 0.0,  # Track total waiting time
    'network_time': 0.0,  # Track total network time
    'count': 0
}



# App-specific latency statistics
app_latency_stats = {}
for app_id in APPLICATIONS:
    app_latency_stats[app_id] = {
        'total_latency': 0.0,
        'propagation_delay': 0.0,
        'spawning_time': 0.0,
        'processing_time': 0.0,
        'waiting_time': 0.0,  # Track app-specific waiting time
        'count': 0
    }

# Function to generate resource demands for an app
# def generate_app_demands(app_id):
#     """Generate CPU and RAM demands for a specific application."""
#     app_config = APPLICATIONS[app_id]
    
#     # Generate warm resource demands
#     cpu_warm = app_config["min_warm_cpu"] if app_config["min_warm_cpu"] == app_config["max_warm_cpu"] else random.uniform(app_config["min_warm_cpu"], app_config["max_warm_cpu"])
#     ram_warm = app_config["min_warm_ram"] if app_config["min_warm_ram"] == app_config["max_warm_ram"] else random.uniform(app_config["min_warm_ram"], app_config["max_warm_ram"])
    
#     # Generate request resource demands
#     cpu_demand = max(cpu_warm, app_config["min_req_cpu"] if app_config["min_req_cpu"] == app_config["max_req_cpu"] else random.uniform(app_config["min_req_cpu"], app_config["max_req_cpu"]))
#     ram_demand = max(ram_warm, app_config["min_req_ram"] if app_config["min_req_ram"] == app_config["max_req_ram"] else random.uniform(app_config["min_req_ram"], app_config["max_req_ram"]))
    
#     bandwidth_direct = app_config["bandwidth_direct"]
#     bandwidth_indirect = app_config["bandwidth_indirect"]

#     generated_resource = {
#         "cpu_warm": cpu_warm,
#         "ram_warm": ram_warm,
#         "cpu_demand": cpu_demand,
#         "ram_demand": ram_demand,
#         "bandwidth_direct": bandwidth_direct,
#         "bandwidth_indirect": bandwidth_indirect
#     }

#     return generated_resource

def generate_app_demands(app_id):
    """Generate CPU and RAM demands for a specific application."""
    app_config = APPLICATIONS[app_id]
    
    # Generate warm resource demands
    # cpu_warm = app_config["min_warm_cpu"] if app_config["min_warm_cpu"] == app_config["max_warm_cpu"] else random.uniform(app_config["min_warm_cpu"], app_config["max_warm_cpu"])
    # ram_warm = app_config["min_warm_ram"] if app_config["min_warm_ram"] == app_config["max_warm_ram"] else random.uniform(app_config["min_warm_ram"], app_config["max_warm_ram"])
    
    # # Generate request resource demands
    # cpu_demand = max(cpu_warm, app_config["min_req_cpu"] if app_config["min_req_cpu"] == app_config["max_req_cpu"] else random.uniform(app_config["min_req_cpu"], app_config["max_req_cpu"]))
    # ram_demand = max(ram_warm, app_config["min_req_ram"] if app_config["min_req_ram"] == app_config["max_req_ram"] else random.uniform(app_config["min_req_ram"], app_config["max_req_ram"]))
    
    # bandwidth_direct = app_config["bandwidth_direct"]
    # bandwidth_indirect = app_config["bandwidth_indirect"]

    generated_resource = {
        "cpu_warm": app_config["min_warm_cpu"],
        "ram_warm": app_config["min_warm_ram"],
        "cpu_demand": app_config["min_req_cpu"],
        "ram_demand": app_config["min_req_ram"],
        # "bandwidth_direct": app_config["bandwidth_direct"],
        # "bandwidth_indirect": app_config["bandwidth_indirect"],
        "packet_size_direct_upload": app_config["packet_size_direct_upload"],
        "packet_size_direct_download": app_config["packet_size_direct_download"],
        "packet_size_indirect_upload": app_config["packet_size_indirect_upload"],
        "packet_size_indirect_download": app_config["packet_size_indirect_download"],
        "data_path_required": app_config["data_path_required"],

    }

    return generated_resource


def log_ave_result():
    """
    Calculate and return simulation metrics.
    """
    # Calculate blocking statistics
    total_blocked = request_stats['blocked_no_server_capacity'] + request_stats['blocked_spawn_failed'] \
                    + request_stats['blocked_no_path']
    
    block_perc = (total_blocked / request_stats['generated'] * 100) if request_stats['generated'] > 0 else 0
    accepted_req = request_stats['generated'] - total_blocked
    # Calculate offloading percentage
    avg_offloaded = (request_stats['offloaded_to_cloud'] * 100 / request_stats['processed']) if request_stats['processed'] > 0 else 0
    
    # Calculate latency metrics
    if latency_stats['count'] > 0:
        avg_total = latency_stats['total_latency'] / latency_stats['count']
        avg_spawn = latency_stats['spawning_time'] / latency_stats['count']
        avg_proc = latency_stats['processing_time'] / latency_stats['count']
        avg_wait = latency_stats['network_time'] / latency_stats['count']
    else:
        avg_total, avg_spawn, avg_proc, avg_wait = 0, 0, 0, 0
    
    # Get a copy of the congested paths dictionary
    congested_paths_dict = congested_paths.copy()
    
    # Prepare results dictionary
    results = {
        'blocking_percentage': float(f"{block_perc:.2f}"),
        'accepted_requests': accepted_req,
        'avg_offloaded_to_cloud': float(f"{avg_offloaded:.2f}"),
        'avg_total_latency': float(f"{avg_total:.3f}"),
        'avg_spawn_time': float(f"{avg_spawn:.3f}"),
        'avg_processing_time': float(f"{avg_proc:.3f}"),
        'avg_network_time': float(f"{avg_wait:.3f}"),
        'congested_paths': congested_paths_dict
    }

    return results
