import random
import math
# --- Configuration ---
RANDOM_SEED = 42
SIM_TIME = 500  # Simulation time units (e.g., seconds)

# Topology configuration
USE_TOPOLOGY = False  # New flag to enable/disable topology routing

# Create a centralized config dictionary to hold all configuration parameters
config = {
    # System Parameters
    "system": {
        "num_servers": 10,
        "sim_time": 1000,
        "distribution": "exponential",  # Options: "deterministic", "uniform", "exponential"
        "verbose": True,
    },
    
    # Server Parameters
    "server": {
        "cpu_capacity": 100.0,  # %
        "ram_capacity": 100.0,  # %
    },
    
    # Request Parameters
    "request": {
        "arrival_rate": 15,  # Average requests per time unit (lambda for M)
        "service_rate": 1,  # Average service completions per time unit (mu for M)
        "bandwidth_demand": 10.0,  # Bandwidth consumption per request
        
        # CPU and RAM demands - fixed values instead of ranges
        "warm_cpu": 1,
        "warm_ram": 10,
        "warm_cpu_model": 1,
        "warm_ram_model": 25,
        "cpu_demand": 30.0,
        "ram_demand": 30.0,
    },
    
    # Container Parameters
    "container": {
        "spawn_time": 4,   # Time units to spawn a container
        "idle_cpu_timeout": 3,     # Time units an idle container waits before removal
        "idle_model_timeout": 0.001,   # Time units an idle model waits before removal
        "load_request_time": 1/1000,  # Time units to load a request into a container
        "load_model_time": 4,    # Time units to load a model into a container
    },
    
    # Topology Parameters
    "topology": {
        "use_topology": False,
    }
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

# New dictionary to track latency metrics (in time units)
latency_stats = {
    'total_latency': 0.0,
    'spawning_time': 0.0,
    'processing_time': 0.0,
    'waiting_time': 0.0,      # Total waiting time (container wait + assignment)
    'container_wait_time': 0.0, # Time waiting for an idle container
    'assignment_time': 0.0,   # Time for the assignment process
    'count': 0
}

# Global reference to topology configuration (for compatibility)
USE_TOPOLOGY = config["topology"]["use_topology"]