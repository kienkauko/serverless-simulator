# Cluster and Server

## Cluster

### 1. Cluster Initiation
A Cluster is primarily defined by two attributes: the network switch node it is attached to and the number of servers it contains.

```python
self.node = config["node"]  # Topology node where this cluster is located
self.servers = [Server(env, self, i, config) for i in range(config["num_servers"])]
```

Each cluster also has `spawn_time_factor` and `processing_time_factor` attributes. These factors adjust the base time required to spawn a container or process a request, reflecting the hardware's performance. For example, a powerful cloud cluster would have **lower** `spawn_time_factor` and `processing_time_factor` values compared to an edge cluster, signifying faster container spawning and request processing.

Other variables are used for continuously recording RAM, CPU, and Energy usage.

### 2. Cluster Functions
The functions within the Cluster class are designed to record and calculate resource utilization. These are called periodically during the simulation to track metrics and at the end of the simulation to provide final results.

## Server

### 1. Server Initiation
Server objects are homogeneous, meaning there is no structural difference between an edge server and a cloud server in the simulation. The distinction lies in their configured attributes. Servers in a cloud cluster are made more powerful by assigning them higher `cpu_capacity` and `ram_capacity`. Similarly, power consumption characteristics (`power_max` and `power_min`) also differ between cloud and edge servers.

### 2. Server Functions
The most critical function in the Server class is `spawn_container_process()`, which handles the creation of new container instances.


