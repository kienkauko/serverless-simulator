# Variables and Metrics

This document clarifies the targeted metrics (Key Performance Indicators - KPIs) and key variables used in the simulator.

---

## Key Performance Indicators (KPIs)

KPIs indicate the performance and consumption of the current system under the current request patterns.

### Performance KPIs

#### Blocking Percentage

Blocking percentage describes the percentage of requests that are successfully served by the system over the total number of generated requests. 

In the current implementation, a request is rejected (failed) only when computing capacity is not enough. Although in `main.py` blocked requests are counted by different types of failures, actually only `request_stats['blocked_no_server_capacity']` is greater than zero.

**Formula:**
```
p_b = (1 - total_blocked/request['generated']) × 100
```

#### Latency

The entire latency perceived by the request (`total_latency`) is the sum of three types of latency:
- **Network delay** (`network_delay`) - latency caused by the network
- **Spawn time** (`spawn_time`) - latency caused by spawning container (if new container needs to be initiated)
- **Processing time** (`processing_time`) - time to process the request

**Formula:**
```
total_latency = request.network_delay + request.spawn_time + request.processing_time
```

At the end of every request's lifecycle, this information is extracted from the request itself and used to calculate the average across all requests:

```python
# Update global latency stats
variables.latency_stats['total_latency'] += total_latency
variables.latency_stats['propagation_delay'] += request.prop_delay
variables.latency_stats['spawning_time'] += request.spawn_time
variables.latency_stats['processing_time'] += request.processing_time
variables.latency_stats['network_time'] += request.network_delay  # Add network time to stats
```

> **Note:** There is another delay called propagation delay (`prop_delay`) but it is already counted in `network_delay`. That's why we don't count it in the `total_latency`.

At the end of simulation, `main.py`/`multi_cases.py` will average out this latency information by dividing it by the number of successful requests.


### Resource Consumption KPIs

We are interested in four consumption metrics: **CPU**, **RAM**, **Power**, and **Network Link Utilization**. 

The first two (CPU and RAM) share the same calculation method. The values from these metrics are the average values at the end of the simulation.

#### CPU and RAM

CPU and RAM usage of the system are the sum of usage across all physical servers. These values change over time when "events" happen in the system, such as:
- Requests arrive
- Requests leave
- Container creation/removal

To capture the time-average value of CPU and RAM:
1. Calculate the sum of CPU and RAM usage area along the time axis
2. Divide this value by the total simulation time

However, capturing CPU/RAM values whenever an event happens would slow down the simulation. Therefore, the simulation captures CPU/RAM instantaneous values at every defined period (changeable via `Cluster.update_period`). This value isn't as accurate as capturing every event, but it's good enough.

At the end of simulation, the average CPU/RAM is queried by calling `Cluster.get_mean_cpu()`:
- If the passing argument is `'cluster'`, the return value will be the average CPU/RAM consumption of the entire cluster
- Otherwise, it will return the average CPU/RAM consumption of servers in that cluster (this is currently not correct, so please don't use it)

#### Power

Power is calculated almost similarly to CPU and RAM. Power of the cluster is the total power of its servers. 

> **Note:** Servers without load will consume 0 Watt, meaning they are OFF. Power calculation only counts servers that are in the ON state.

#### Link Utilization

TBD

---

## Variables

### Simulation Variables

The most important variable is `SIM_TIME`. This indicates the simulation time of the simulator (could be second, minute, hour, etc., depending on your interpretation). 

> **Important:** The higher `SIM_TIME` is, the longer you have to wait for the simulation to finish. It's better to keep it at a reasonable value.


### Topology Variables

There are many variables, but the most important are:

#### `CLUSTER_STRATEGY`
Indicates the edge-cloud strategies for DC placement as well as request forwarding. Currently there are three options:

- **`centralized_cloud`** - Only one DC acts as the central cloud. All requests must send data there.
- **`massive_edge_cloud`** - Besides cloud DC, edge DCs also exist. Requests are forwarded to edge DCs in proximity first. If resources at the edge are not enough, requests must send data to the cloud.
- **`massive_edge`** - Only edge DCs exist. Requests send data to nearby edge DCs. If their DC runs out of resources, then requests are rejected.

#### `EDGE_SERVER_NUMBER`
Indicates the total number of edge servers that are distributed geographically over the topology.

#### `EDGE_DC_LEVEL`
Denotes at which switch layer edge DCs are connected to/situated at. 

For example, if we set this variable equal to 2, then there will be an edge DC at each layer-2 switch.

#### `EDGE_SERVER_PROVISION_STRATEGY`
Denotes how edge servers are distributed over the edge DCs. There are two options:

- **`equally`** - Edge servers are distributed equally over edge DCs
- **`population`** - Edge servers are distributed following the population of the edge DCs. Areas with higher population will have more edge servers.

#### `CLOUD_SPAWN_TIME_FACTOR`
Controls how fast a cloud server can spawn a container. Since cloud servers are typically more powerful than edge servers, they can spawn containers faster. 

As `Server` in this simulator is a homogeneous class, this variable is introduced to indicate that cloud servers are faster than edge servers. 

**Example:** A value of `0.6` means spawning a container on a cloud server takes only 60% of the spawning time compared to an edge server.

#### `CLOUD_PROCESSING_TIME_FACTOR`
Similar to the spawn time factor, it indicates how fast cloud servers can process requests compared to edge servers.

---

### Application, Request, Container Variables

Applications' profiles are defined in the `APPLICATIONS` dictionary.

The simulation allows multiple heterogeneous apps to be simulated, but currently only one app can work correctly. 

For each application, there are some important notes about its variables:

#### CPU and RAM Usage
- `cpu` and `ram` usage have `min` and `max` values
- This indicates the range in which they potentially consume
- **Example:** `min_warm_cpu = 5` and `max_warm_cpu = 7` means this application during warm state typically consumes in the range of 5 to 7% CPU
- The function `generate_app_demands()` will generate a random consumption within this range to ensure statistical behavior for the app
- However, the current running has turned this function off, which is why both min and max have the same value

#### Data Location
- `data_location` is the location of data
- We allow applications to query data from a different place
- **Example:** A video encoder application can be placed at an edge server while pulling raw video (data) from a cloud server
- However, there are applications that do not need data from other locations
- By changing `data_path_required` to `True` or `False`, we can specify if the application needs data from elsewhere or not

#### Packet Sizes
- **`packet_size_direct_upload`** - The data file size sent from the user
- **`packet_size_direct_download`** - The result data sent back to the user once processing finishes
- **`packet_size_indirect_upload`** and **`packet_size_indirect_download`** - Used only when data location is required (`data_path_required` is `True`)

---

### Data Recording Variables


