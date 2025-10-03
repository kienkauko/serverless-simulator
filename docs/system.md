# System (`System.py`)

The `System` class serves as the main orchestrator for the simulation, coordinating servers, requests, and containers across the network topology. It manages the complete lifecycle of requests from generation to completion.

## 1. System Initialization

System initialization involves setting up the core components and infrastructure for the simulation:

### Core Components Setup
- Creates SimPy environment for discrete event simulation
- Links to the network topology instance containing clusters and routing information
- Initializes request ID counter using `itertools.count()` for unique request identification

### Scheduler and Container Management
- Instantiates schedulers for each cluster using the specified scheduler class (default: `FirstFitScheduler`)
- Creates idle container pools (`app_idle_containers`) organized by cluster and application type
- Configures idle timeout values for applications across clusters using `variables.UNIVERSAL_TIMEOUT`

### Load Balancer Integration
- Initializes a centralized `LoadBalancer` that manages request distribution across multiple clusters
- Load balancer receives references to all cluster schedulers for coordination

The system supports multiple cluster strategies (centralized cloud, distributed cloud, massive edge) and automatically adapts container management to the chosen topology configuration.

## 2. Request Generation

The system generates requests asynchronously for multiple applications using Poisson arrival processes:

### Multi-Application Request Generation
- `request_generator()` initiates request generation for all defined applications in `variables.APPLICATIONS`
- Each application runs its own independent request generation process via `app_request_generator()`
- Applications can generate requests simultaneously without interference, enabling realistic multi-tenant scenarios

### Geographic Request Distribution
- Only level 3 nodes (edge nodes) in the network topology can generate requests
- Node selection is controlled by `node_intensity` parameter (percentage probability)
- Request arrival rate for each node is calculated as: `node_population × variables.TRAFFIC_INTENSITY`

### Poisson Process Implementation
- Inter-arrival times follow exponential distribution using `random.expovariate(arrival_rate)`
- Each request receives unique ID, timestamp, and resource demands specific to its application
- Resource demands are generated using `variables.generate_app_demands(app_id)` for application-specific requirements

### Request Characteristics
- Requests include origin node, data location requirements, and application-specific configurations
- System tracks total expected arrival rate across all nodes and applications
- Each generated request triggers an asynchronous `handle_request()` process for immediate processing

## 3. Request Handler

The `handle_request()` method manages the complete lifecycle of incoming requests through several coordinated phases:

1. **Path Discovery Phase:**
   - Calls `topology.find_cluster(request)` to identify viable clusters based on current strategy
   - Returns dictionary of target clusters with their respective network paths `[path_direct, path_indirect]`
   - If no network paths are available, request fails with 'link_failed' status and detailed failure statistics by network level

2. **Load Balancing and Resource Allocation:**
   - Delegates request to `LoadBalancer.handle_request()` with viable cluster options
   - Load balancer evaluates clusters and attempts container assignment
   - Returns assignment result, assigned container, and selected cluster

3. **Network Path Implementation (Success Path):**
   - Implements network paths using `topology.make_paths()` for bandwidth reservation or flow counting
   - Calculates upload delay using `topology.update_request_delay()` with 'upload' type
   - Network delay includes both propagation delay and transmission delay (in PS mode)

4. **Service Execution:**
   - Initiates container service lifecycle via `container.service_lifecycle()` 
   - Container handles request processing, resource allocation, and timing
   - Calculates download delay after service completion

5. **Cleanup and Statistics:**
   - Updates comprehensive statistics via `update_end_statistics()` for successful requests
   - Releases container resources and returns container to idle pool for reuse
   - Removes network paths using `topology.remove_paths()` to free bandwidth/flows
   - Handles failure cases with appropriate error statistics (compute_failed, link_failed)

### Failure Handling
- **Network failures** tracked by link level (3-3, 3-2, 2-2, etc.) for congestion analysis
- **Compute failures** recorded when no server capacity available
- All failures include detailed breakdown for performance optimization