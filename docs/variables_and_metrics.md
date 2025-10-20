This doc clarifies the targeted metrics (key performance indicators - KPIs) and some key variables used the the simulators.

## KPIs

KPIs indicate the performance and consumption of the current system under the current request patterns

The performance KPIs:

**Blocking percentage**: blocking percantage describes the percentage of requests that are successfully served by the system over the total number of generated requests. In the current implementation, a request is rejected (failed) only when computing capacity is not enough (eventhough in main.py blocked_requests are counted by different types of failures, but actually only request_stats['blocked_no_server_capacity'] is greater than zero) To this end, blocking percentange is defined as:
p_b = (1 - total_blocked/request['generated']) * 100

**Latency**: the entire latency perceived by the request (total_latency) is the sum of three types of latency: latency caused by the network (network_delay), latency caused by spawning container (if new container needs to be initiated) (spawn_time) and processing time (processing_time) to process the request.
total_latency = request.network_delay + request.spawn_time + request.processing_time

At the end of every request's lifecycle, these information are extracted from the request itself, they are then used to calculate the average out of every request

# Update global latency stats
variables.latency_stats['total_latency'] += total_latency
variables.latency_stats['propagation_delay'] += request.prop_delay
variables.latency_stats['spawning_time'] += request.spawn_time
variables.latency_stats['processing_time'] += request.processing_time
variables.latency_stats['network_time'] += request.network_delay  # Add network time to stats

Note: there is another delay called propagation delay (prop_delay) but it is counted in network_delay already. That's why we don't count it in the total_latency.

At the end of simulation, the main.py/multi_cases.py will average out these latency infomation by dividing them for the number of successed requests.

The resource consumption KPIs:

We are interested in four consumption metrics, they are, CPU, RAM, Power and network link utilization. The first two (CPU and RAM) shares the same way of calculation. The values out of these metrics are the average value at the end of the simulation. 

**CPU and RAM**:  CPU and RAM usage of the system are sum of that of all physical servers. These values change overtime when there are 'events' happening in the system, such as requests arrive, requests leave, container creation/removal, etc. To capture the time average value of CPU and RAM, we first need to calculate the sum of CPU and RAM usage area along with time, then we divide this value by the total simulation time. (Figure 1). However, if we capture the CPU/RAM value whenever an event happens will slow down the simulation. Thus, the simulation capture the CPU/RAM instantaneous value at every defined period of time (changeable via Cluster.update_period), this value for sure isn't as correct as if we capture every event, but it's good enough (Figure 2). At the end of simulation, the average CPU/RAM is querried by calling function: Cluster.get_mean_cpu(). If the passing argument is 'cluster', the return value will be the average CPU/RAM consumption of the entire cluster. Otherwise, it will return the average CPU/RAM consumption of server in that cluster (this is currently not correct, so please don't use it). 

**Power**: Power is calculated almost similar to CPU and RAM. Power of cluster is the total power of its servers. It is noted that server without load will consume 0 Watt, meaning it is OFF. Power calculation only counts servers that are at ON state.

**Link Utilization**: TBD

## Varibles
Simulation variables: The most important variable is SIM_TIME. This indicates the simulation time of the simulator. It could be second, minute, hour, etc. depending on how do you want to understand it. Just keep in mind that the higher SIM_TIME is, the longer you have to wait for the simulation to finish. So it is better to leave it be.

Topology variables: There are many variables, but the most important are:
CLUSTER_STRATEGY
EDGE_SERVER_NUMBER 
EDGE_DC_LEVEL 
EDGE_SERVER_PROVISION_STRATEGY 
CLOUD_SPAWN_TIME_FACTOR 
CLOUD_PROCESSING_TIME_FACTOR 

Application, Request, Container variables:

Data recording variables:
