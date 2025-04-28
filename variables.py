import random
# --- Configuration ---
RANDOM_SEED = 42
SIM_TIME = 500  # Simulation time units (e.g., seconds)


# Container Parameters
CONTAINER_ASSIGN_RATE = 1000.0 # Average rate for request assignment (very fast)

# Topology configuration
USE_TOPOLOGY = False  # New flag to enable/disable topology routing

# --- Multi-Cluster Configuration ---
# Define the parameters for each cluster
CLUSTER_CONFIG = {
    "edge": {
        "node": "nodeA",
        "num_servers": 2,
        "server_cpu": 100.0,
        "server_ram": 100.0,
        "spawn_time_factor": 1.0  # Edge spawn time multiplier (slower)
    },
    # "cloud": {
    #     "node": "nodeB",
    #     "num_servers": 4,
    #     "server_cpu": 200.0,
    #     "server_ram": 200.0,
    #     "spawn_time_factor": 0.7  # Cloud spawn time multiplier (faster)
    # }
}

# System Parameters
# NUM_SERVERS = 2
# SERVER_CPU_CAPACITY = 100.0 # %
# SERVER_RAM_CAPACITY = 100.0 # %

# Application definitions for heterogeneous workloads
APPLICATIONS = {
    "app1": {
        "arrival_rate": 5.0,  # Requests per time unit
        "service_rate": 2.0,  # Service completions per time unit
        "base_spawn_time": 5.0,  # Base time units to spawn a container (modified by cluster factor)
        "min_warm_cpu": 0.5,  # Minimum CPU for warm container
        "max_warm_cpu": 0.5,  # Maximum CPU for warm container
        "min_warm_ram": 5.0,  # Minimum RAM for warm container
        "max_warm_ram": 5.0, # Maximum RAM for warm container
        "min_req_cpu": 50.0,  # Minimum CPU demand for request
        "max_req_cpu": 50.0,  # Maximum CPU demand for request
        "min_req_ram": 20.0,  # Minimum RAM demand for request
        "max_req_ram": 20.0,  # Maximum RAM demand for request
        "bandwidth_demand": 5.0,  # Bandwidth demand for this application
    },
    "app2": {
        "arrival_rate": 3.0,
        "service_rate": 1.5,
        "base_spawn_time": 7.0,
        "min_warm_cpu": 1.0,
        "max_warm_cpu": 1.0,
        "min_warm_ram": 8.0,
        "max_warm_ram": 8.0,
        "min_req_cpu": 40.0,
        "max_req_cpu": 40.0,
        "min_req_ram": 40.0,
        "max_req_ram": 40.0,
        "bandwidth_demand": 10.0,  # Bandwidth demand for this application
    },
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
    #     "bandwidth_demand": 15.0,  # Bandwidth demand for this application
    # }
}

# Statistics
request_stats = {
    'generated': 0,
    'processed': 0,
    'blocked_no_server_capacity': 0, # Blocked because no server could *ever* fit it
    'blocked_spawn_failed': 0,      # Blocked because spawning failed (transient lack of resources)
    'blocked_no_path': 0,  # New: rejected due to no routing path with available bandwidth
    'container_spawns_initiated': 0,
    'container_spawns_failed': 0,
    'container_spawns_succeeded': 0,
    'containers_reused': 0,
    'containers_removed_idle': 0,
    'reuse_oom_failures': 0, # Out Of Memory/CPU when trying to activate reused container
}

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
    }

# New dictionary to track latency metrics (in time units)
latency_stats = {
    'total_latency': 0.0,
    'propagation_delay': 0.0,
    'spawning_time': 0.0,
    'processing_time': 0.0,
    'waiting_time': 0.0,  # Track total waiting time
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
def generate_app_demands(app_id):
    """Generate CPU and RAM demands for a specific application."""
    app_config = APPLICATIONS[app_id]
    
    # Generate warm resource demands
    cpu_warm = app_config["min_warm_cpu"] if app_config["min_warm_cpu"] == app_config["max_warm_cpu"] else random.uniform(app_config["min_warm_cpu"], app_config["max_warm_cpu"])
    ram_warm = app_config["min_warm_ram"] if app_config["min_warm_ram"] == app_config["max_warm_ram"] else random.uniform(app_config["min_warm_ram"], app_config["max_warm_ram"])
    
    # Generate request resource demands
    cpu_demand = max(cpu_warm, app_config["min_req_cpu"] if app_config["min_req_cpu"] == app_config["max_req_cpu"] else random.uniform(app_config["min_req_cpu"], app_config["max_req_cpu"]))
    ram_demand = max(ram_warm, app_config["min_req_ram"] if app_config["min_req_ram"] == app_config["max_req_ram"] else random.uniform(app_config["min_req_ram"], app_config["max_req_ram"]))
    
    return cpu_warm, ram_warm, cpu_demand, ram_demand
