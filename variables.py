import random
# --- Configuration ---
RANDOM_SEED = 42
SIM_TIME = 500  # Simulation time units (e.g., seconds)

# Topology configuration
USE_TOPOLOGY = False  # New flag to enable/disable topology routing

# System Parameters
NUM_SERVERS = 2
SERVER_CPU_CAPACITY = 100.0 # %
SERVER_RAM_CAPACITY = 100.0 # %

# Request Parameters
REQUEST_ARRIVAL_RATE = 10  # Average requests per time unit (lambda for M)
REQUEST_SERVICE_RATE = 1  # Average service completions per time unit (mu for M)

# Request resource demands (example: uniform distribution)
MIN_WARM_CPU = 0.5
MAX_WARM_CPU = 2.0
MIN_WARM_RAM = 5.0
MAX_WARM_RAM = 10.0

# Request resource demands (example: uniform distribution)
MIN_REQ_CPU = 49.0
MAX_REQ_CPU = 49.0
MIN_REQ_RAM = 49.0
MAX_REQ_RAM = 49.0

# New constant for bandwidth consumption per request
BANDWIDTH_DEMAND = 10.0

# Container Parameters
CONTAINER_SPAWN_TIME = 5   # Time units to spawn a container
CONTAINER_IDLE_TIMEOUT = 2 # Time units an idle container waits before removal
CONTAINER_ASSIGN_RATE = 1000.0 # Average rate for request assignment (very fast)

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
    'propagation_delay': 0.0,
    'spawning_time': 0.0,
    'processing_time': 0.0,
    'waiting_time': 0.0,      # Total waiting time (container wait + assignment)
    'container_wait_time': 0.0, # Time waiting for an idle container
    'assignment_time': 0.0,   # Time for the assignment process
    'count': 0
}

# Define resource demands for requests
"""Generate CPU and RAM demands for a warm request."""
CPU_WARM = MIN_WARM_CPU if MIN_WARM_CPU == MAX_WARM_CPU else random.uniform(MIN_WARM_CPU, MAX_WARM_CPU)
RAM_WARM = MIN_WARM_RAM if MIN_WARM_RAM == MAX_WARM_RAM else random.uniform(MIN_WARM_RAM, MAX_WARM_RAM)

"""Generate CPU and RAM demands for a request, ensuring they are >= warm demands."""
CPU_DEMAND = max(CPU_WARM, MIN_REQ_CPU if MIN_REQ_CPU == MAX_REQ_CPU else random.uniform(MIN_REQ_CPU, MAX_REQ_CPU))
RAM_DEMAND = max(RAM_WARM, MIN_REQ_RAM if MIN_REQ_RAM == MAX_REQ_RAM else random.uniform(MIN_REQ_RAM, MAX_REQ_RAM))