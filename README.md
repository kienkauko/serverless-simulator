# Detailed Simulation Report

- **Report Generated:** 10:43AM on 16/04/2025
- **Updated:** 09/09/2025

## Simulator Functions & Mechanisms

- **Request Generation & Routing:**
  - Requests are generated at level 3 nodes based on population distribution.
  - Each request is routed from its origin node to the compute cluster using a hierarchical network topology.
  - The topology reads edge attributes (bandwidth, mean latency, jitter).
  - **Path Latency Generation:**  
    - For each edge, effective latency = max(0, random.gauss(mean_latency, jitter)).
    - The round-trip propagation delay is calculated as 2 × (sum of latencies along the path between user and compute node only).
  - Only one path is chosen per routing decision. If this path has insufficient bandwidth, the request is rejected.

- **Total Latency Calculation:**
  - Total latency is computed as:  
    Total Latency = Propagation Delay + Container Spawn Time + Service Processing Time.

- **Load Balancer and Scheduler:**
  - The system has been refactored to include separate Load Balancer and Scheduler components:
    - **Load Balancer (LB):** Receives incoming requests and determines which container instances should handle them.
      - Currently implements a random selection strategy for idle containers.
      - Functions as a local LB, limited to managing instances within a single cluster.
    - **Scheduler:** Responsible for spawning new containers when needed.
      - Implements two strategies: FirstFit and BestFit for container placement.
      - Currently operates within a single cluster scope.

- **Container Assignment Mechanism:**
  - On request arrival, the request is directed to the Load Balancer:
    - The LB checks for available idle containers using its selection strategy.
    - If an idle container is found, the request is immediately forwarded to it.
    - If no suitable container is available, the LB calls the Scheduler to spawn a new container.
  - Idle containers are monitored with a timeout; if reused before expiration, the idle timeout is cancelled.

- **Resource Consumption at Each State:**
  - **Before Request Assignment:**
    - Servers maintain two resources: "real" (warm resources) and "reserve" to reserve resources for "active" state of container when a request is assigned to it.
  - **When Assigning a Request:**
    - Container resources are reserved with Active consumption rates.
    - If the request demands exceed available resources, the request is rejected.
  - **During Request Processing:**
    - The container remains active while consuming allocated resources.
  - **After Processing:**
    - The container reverts to its warm state and is marked idle.
    - Idle containers are placed in an idle pool for potential reuse.

## Request and Container Characteristics

- **Request Characteristics:**
  - CPU and RAM requirements for both warm and active states
  - Bandwidth requirements:
    - Direct bandwidth (from user to compute)
    - Indirect bandwidth (from data to compute)
  - Data node location (where data is stored)
  - Origin node (where the request is generated)

- **Container States:**
  - Three distinct states: Null, Warm, and Active
  - Each application type can have its own idle timeout per cluster (currently set to 2.0 time units)

## Multi-Application Support

- **Multiple Application Types:**
  - The simulator supports simultaneous processing of multiple application types.
  - Each app type can have different resource requirements and processing characteristics.
  - Performance metrics are tracked separately for each application type.

## Collected Metrics

- **Currently Implemented Metrics:**
  - Acceptance rate of each application
  - Rejection reasons categorized by:
    - Link bandwidth limitations
    - Compute resource limitations
  - Cluster resource utilization (CPU, RAM)
  - Cluster energy consumption
  - Blocked requests based on link level

- **Metrics Management:**
  - Resource statistics (CPU, RAM, energy) are calculated per cluster in the Cluster class
  - Metrics are presented at runtime in the main file
  - Currently, metrics are not persistently stored between runs

## Topology Implementation

- **Germany Network Topology:**
  - Hierarchical network topology of Germany has been implemented
  - Paths are routed via hierarchical path finding algorithm
  - Node levels are incorporated into the topology structure
  - Population data at level 3 nodes determines request generation distribution

## Runtime Optimization

- **Performance Improvements:**
  - Implemented hierarchical path finding to reduce computational overhead
  - Added handling for special cases:
    - Path finding when source and destination are the same (data link)
    - Path finding between different-level sources and destinations
  - Path caching implemented to reduce repeated path calculations

## Future Development (TODOs)

- **Topology Improvements:**
  - Implement alternative path selection when primary path has insufficient bandwidth
  - Analyze link utilization based on link level (3-3, 3-2, 2-2, 2-1, etc.) to identify bottlenecks

- **Code Structure Improvements:**
  - Standardize usage of Cluster and Cluster_name references
  - Refine Container class to reduce dependency chains (Container → Server → Cluster)
  - Consolidate metrics collection (currently split between System, Variables, and Cluster classes)

- **Features to Consider:**
  - Better differentiation between edge and cloud computing beyond scaling resource capacities
  - Allow each cluster to have its own scheduler implementation
