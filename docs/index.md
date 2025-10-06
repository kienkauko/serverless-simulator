
# Serverless Simulator Documentation

## Prerequisites

Before using the simulator, ensure you understand:
- **Serverless computing** fundamentals and concepts
- **Simpy** event simulation
- (Optional) **Topology paper** and play around with **topology visualization** found in [/docs/topology_demo](/docs/topology_demo) folder. (Just read the paper briefly to understand the topology)

## Main File Functions

### Core Simulation Files
1. **`main.py`** - Entry point that creates topology, system, starts simulation and prints final results
2. **`multi_cases.py`** - Extended version of main.py with loops to run multiple simulation scenarios and save results to Excel

### Core Components
3. **`Topology.py`** - Creates network topology from input files, defines datacenter locations, resource capacity, and routing algorithms (see [topology.md](topology.md) for details)
4. **`System.py`** - Generates and manages requests by coordinating with other components
5. **`Request.py`** - Request object containing all request information and properties
6. **`LoadBalancer.py`** - Routes requests to appropriate clusters using defined strategies
7. **`Scheduler.py`** - Creates containers on specific servers within clusters using scheduling strategies
8. **`Cluster.py`** - Represents datacenters at specific network nodes, containing multiple servers
9. **`Server.py`** - Individual server objects with resource management capabilities
10. **`Container.py`** - Container objects with lifecycle management and resource allocation/deallocation functions
11. **`variables.py`** - Global variables for recording system metrics throughout the simulation

## Code Flow Overview

Starting from `main.py`, the simulation follows this sequence:

1. **Initialize** SimPy environment
2. **Create** network topology
3. **Initialize** system components
4. **Generate** requests via `system.request_generator()`
5. **Handle** requests via `system.handle_request()` (calls other functions as needed)
6. **Complete** simulation when environment timeout reached
7. **Output** tracked metrics and results

> **Note:** Detailed request handling flow can be found in `request_handler.md`

## Documentation Index

| Document | Description |
|----------|-------------|
| **[topology.md](topology.md)** | Network topology creation, datacenter placement, and routing algorithms between network nodes |
| **[system.md](system.md)** | Main simulation orchestrator that generates requests, coordinates components, and manages request lifecycle |
| **[LB_and_scheduler.md](LB_and_scheduler.md)** | Load balancer request distribution and server scheduling strategies for container placement |
| **[cluster_and_server.md](cluster_and_server.md)** | Datacenter clusters and individual server resource management and container spawning |
| **[container.md](container.md)** | Container lifecycle, resource allocation/deallocation, and state management (idle, active, dead) |
| **[strategies.md](strategies.md)** | Comprehensive guide to all placement, routing, scheduling, and timeout strategies available in the simulator |

## Last Notes  

Function with (`deprecated`) means it is not used anymore, please ignore it.

