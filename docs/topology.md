# Topology (`Topology.py`)

The `Topology` class manages network infrastructure, datacenter placement, and routing algorithms for the serverless simulation environment. It creates a hierarchical network topology based on real geographic data and implements sophisticated routing mechanisms.

## 1. Topology Initialization

The topology is initialized through the `Topology` class constructor with the following steps:

### Graph Construction
- Creates a directed graph using NetworkX to represent the network topology
- Loads network nodes and links from JSON files (`edge_path` parameter)
- Each node contains attributes: location (EPSG:3857 coordinates), level (hierarchical level 0-3), population, and parent node
- **Note:** Level 3 nodes are used as ingress nodes (where requests originate). Population data for level 3 nodes simulates user traffic patterns
- Links between nodes include bandwidth capacity, latency (calculated using Haversine formula based on geographic distance), and level information

> *More information about topology structure is found in section [Topology JSON File Structure](#3-topology-json-file-structure)*

### Cluster Initialization
After switch nodes and edges (links between nodes) are created, datacenters (either edge DC or cloud DC) are initiated and placed over the newly-created topology.

> *More details about datacenter placement can be found in [strategies.md](strategies.md)*

- Loads cluster configurations from JSON file (`cluster_info` parameter)
- Supports different cluster strategies via `variables.CLUSTER_STRATEGY`:
  - `"centralized_cloud"`: Single central cloud cluster
  - `"distributed_cloud"`: Multiple distributed cloud clusters (deprecated)
  - `"massive_edge"` or `"massive_edge_cloud"`: Edge clusters at specified network levels
- Assigns nearby clusters for ingress nodes (level 3 nodes), meaning requests from an ingress node can only be assigned to its nearby clusters:

```python
for node_id in [n for n, data in self.graph.nodes(data=True) if data.get('level') == 3]:
    nearby_clusters = self.get_nearby(variables.CLUSTER_STRATEGY, node_id, variables.EDGE_DC_LEVEL)
    self.graph.nodes[node_id]['nearby_clusters'] = nearby_clusters
```

## 2. Cluster Discovery and Network Routing

### Path to Cluster Discovery (`find_cluster()`)
Since each request (originates from an Ingress node) has information about its nearby cluster. This function simply finds routes to the chosen clusters for request:
- **Centralized Cloud**: Directly routes to the single central cloud node defined by `variables.CENTRAL_CLOUD`
- **Distributed Strategies**: Uses pre-calculated nearby clusters for the request's origin node
- Returns a dictionary mapping clusters to their respective paths `[path_direct, path_indirect]` and any failed link information
- For each potential cluster, validates both direct path (origin→cluster) and indirect path (cluster→data) if data access is required

> **Note:** Why are there two paths (direct and indirect)? Please read the request definition in `request.md`.

### Path Finding Between Nodes (`find_possible_path()`)
This function is fundamental and is called whenever route between two nodes are required. For example, the above `find_cluster()` function also calls this function. Hierarchical routing is done based on network topology levels:

- **Same Level Routing**: 
  - Level 0 nodes: Direct connection or single hop
  - Higher levels: Routes through common parent nodes, escalating to grandparents and great-grandparents as needed
- **Cross-Level Routing**: Routes from higher-level node to its parent until reaching the target level, then connects to destination
- **Path Caching**: Uses `path_cache` dictionary to store computed paths with keys `(src, dst)` for performance optimization
- **Bandwidth Validation**: In reservation mode, checks available bandwidth; in PS mode, always allows path establishment
- Returns success status, path, and failed link information grouped by network level

### Path Implementation and Latency Calculation

**Path Implementation (`make_paths()`, `remove_paths()`):**
- **Reservation Mode** (deprecated): Reserves/releases specific bandwidth amounts along path edges
- **Packet Switching (PS) Mode**: Increments/decrements active flow counters on path edges without bandwidth reservation

**Latency Calculation in PS Mode:**
- **Propagation Delay**: Static delay based on geographic distance using cached path latency calculations
- **Transmission Delay**: Dynamic delay calculated using bottleneck link methodology:
  - For each link: `flow_bandwidth = link_capacity / (num_active_flows + 1)`
  - Bottleneck bandwidth = minimum flow bandwidth across all path links
  - Total transmission delay = `packet_size / bottleneck_bandwidth`
  - Models pipelining by using single bottleneck rather than summing individual link delays
- **Total Network Delay**: Sum of propagation delay and transmission delays for both direct and indirect paths
- The system tracks bottleneck link levels for congestion analysis and performance monitoring

> **Note:** Reservation mode performs simple bandwidth checking but is deprecated and not actively used in current simulations.

## 3. Topology JSON File Structure
Information about topology is found in `edge.json` file. This file describes a pseudo network topology of Germany. The structure is:

```json
{
    "nodes": [
        {
            "name": "1_R0",
            "type": "switch",
            "location": [
                1460286.4296253233,
                7228753.191889011
            ],
            "level": 3,
            "population": 402.761387896931,
            "parent": 12479
        }
        // ...more nodes...
    ],
    "links": [
        {
            "n1": "12876_R0",
            "n2": "12877_R0",
            "bandwidth": 400000000000.0
        }
        // ...more links...
    ]
}
```

- **`nodes`**: Information about switch nodes
- **`links`**: Information about physical links connecting two switch nodes

Switch nodes have `level` attributes that indicate their hierarchical position. The higher the level, the closer it is to users (like switches of your university or neighborhood). Lower level switches are "bigger" and responsible for higher level switches, creating a hierarchical network structure.

The `parent` attribute indicates the gateway for traffic that needs to leave the local network. For example, traffic from a level-3 switch may need to traverse to its parent (a level-2 switch) to reach a distant destination. Traffic may need to go to lower level switches (bigger and core switches) to reach its destination.

> To understand more about the topology and experiment with it, refer to the associated research paper and run the simulation scripts in `/docs/topology_demo`.

## 4. Performance Optimization
### Caching
There are some functions with 'cache' name such as `get_cached_path()` and `save_cached_path()`. These are used to cache found routes between any node A and B and the propagation latency between them so that later when a route between A and B is requested, we can take it out directly from the cache information instead of running `find_possible_path()` again. This helps reducing runtime of the simulation especially when a large number of requests is simulated. 

