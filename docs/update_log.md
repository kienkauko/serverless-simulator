# Update Log

## Updated 25/10/2025
**Commit:** []()

### Changes

- `multi_cases` has been updated with to adapt to all the changes
- Some functions' name in `System.py` and `Container.py` are changed.
- Update topologies of some selected countries in [/topology/countries](../topology/countries)  folder.
  

## Updated 21/10/2025
**Commit:** [31f1d3e](https://github.com/kienkauko/serverless-simulator/commit/31f1d3ed2870d3a310b969e42020a6360d713638)

### Bug Fix

- **Fixed network flow lifecycle management:** Previously, network connections between ingress nodes and containers were kept active for the entire request processing duration. This incorrectly maintained flow counts even after file uploads/downloads completed, causing inaccurate latency calculations. The updated implementation now establishes separate connections for each "upload" and "download" phase:
  1. Establish connection
  2. Transfer data and calculate latency
  3. Wait for transmission time (simpy timeout)
  4. Close connection and remove flow
  
  See changes in [`System.py`](../System.py) and [`Topology.update_request_delay()`](../Topology.py)

### Changes

- Refactored TCP delay calculation in [`Topology.update_request_delay()`](../Topology.py) to support phased connection management
- Merged `make_paths()` and `implement_path()` into single function for improved code organization
- Merged `remove_paths()` and `release_path()` into single function (matching above pattern)
- Removed `reservation` option for Topology, now only `ps` is available.
  


## Updated 20/10/2025

**Commit:** [dbc0bb4](https://github.com/kienkauko/serverless-simulator/commit/dbc0bb496f4df9e7ad0c96ac72a22b04a4de29b5)

### Changes

- Updated documentation: [variables_and_metrics.md](/docs/variables_and_metrics.md) (work in progress)
- Added `log_result()` function in [variables.py](/variables.py) to centralize KPI calculations (blocking rate, latency, etc.), previously scattered in main or multi_cases files
- **BREAKING:** Modified traffic generation behavior - traffic can now be limited to specific layer-3 nodes instead of all nodes. See `define_ingress_nodes()` in [Topology.py](/Topology.py) for details. Currently supports filtering by city
- Added `LINK_UTILIZATION = {}` metric in [variables.py](/variables.py) to track initial link utilization representing fixed baseline traffic (e.g., `0.2` = 20% of link capacity occupied)

---

## Updated 10/10/2025

**Commit:** [5e99154](https://github.com/kienkauko/serverless-simulator/commit/5e991544ed564846fc6403089e21b6c0776b670a)

### Changes

- **REMOVED:** `distributed_cloud` placement strategy
- **REMOVED:** `cluster.json` file (no longer needed for distributed cloud DC specification)
- **ADDED:** DC specification in [edge.json](/topology/edge.json). Example for central cloud:
  ```json
  {
      "name": "central_cloud", 
      "node": "12876",
      "num_servers": 15000, 
      "server_cpu": 200.0,
      "server_ram": 200.0,
      "power_max": 150,
      "power_min": 50,
      "spawn_time_factor": 0.5,
      "processing_time_factor": 0.6
  }
  ```
- Code cleanup reflecting above architectural