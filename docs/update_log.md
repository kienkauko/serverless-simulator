# Update Log

## Updated 20/10/2025

**Commit:** [5e99154](https://github.com/kienkauko/serverless-simulator/commit/5e991544ed564846fc6403089e21b6c0776b670a#diff-2f85b188f74cf75b146c6806d50231d217aaf6f6064b246de59cd949836335f2)

### Changes

- Updated documentation: [variables_and_metrics.md](/docs/variables_and_metrics.md) (work in progress)
- Added `log_result()` function in [variables.py](/variables.py) to centralize KPI calculations (blocking rate, latency, etc.), previously scattered in main or multi_cases files
- **BREAKING:** Modified traffic generation behavior - traffic can now be limited to specific layer-3 nodes instead of all nodes. See `define_ingress_nodes()` in [Topology.py](/Topology.py) for details. Currently supports filtering by city
- Added `LINK_UTILIZATION = {}` metric in [variables.py](/variables.py) to track initial link utilization representing fixed baseline traffic (e.g., `0.2` = 20% of link capacity occupied)

---

## Updated 10/10/2025

**Commit:** [5e99154](https://github.com/kienkauko/serverless-simulator/commit/5e991544ed564846fc6403089e21b6c0776b670a#diff-2f85b188f74cf75b146c6806d50231d217aaf6f6064b246de59cd949836335f2)

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