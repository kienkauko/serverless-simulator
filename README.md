# Detailed Simulation Report

- **Report Generated:** 10:43AM on 16/04/2025
- **Updated:** 25/04/2025

## Simulator Functions & Mechanisms

- **Request Generation & Routing:**
  - Requests are created randomly at nodes.
  - Each request is routed from its origin node to the cluster using a network topology.
  - The topology reads edge attributes (bandwidth, mean latency, jitter).
  - **Path Latency Generation:**  
    - For each edge, effective latency = max(0, random.gauss(mean_latency, jitter)).
    - The round-trip propagation delay is calculated as 2 × (sum of latencies along the path).

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
    - Initially, a container uses its warm resource allocation.
    - If the request demands exceed the container's warm allocation, additional resources are drawn from the server's reserve.
  - **During Request Processing:**
    - The container remains active while consuming additional allocated resources.
  - **After Processing:**
    - Extra allocated resources are released; the container reverts to its default warm state and is marked idle.
    - Idle containers are then placed in an idle pool for potential reuse.

- **Additional Simulator Notes:**
  - Simulation statistics aggregate metrics such as processed requests, resource spawning attempts, blocking reasons, and latency sums.
  - Detailed logging is implemented to track each step in the lifecycle of requests and containers.

## Multi-Application Support

- **Multiple Application Types:**
  - The simulator now supports simultaneous processing of multiple application types.
  - Each app type can have different resource requirements and processing characteristics.
  - Performance metrics are tracked separately for each application type.

- **Current Limitations:**
  - Container idle timeout is currently fixed across all applications.
  - Future implementations will support app-specific timeout configurations.

## Future Development (TODOs)

- **Cluster Selection Strategies:**
  - Expand Load Balancer strategies to support selection between multiple clusters.
  - Implement multi-cluster scheduling strategies in the Scheduler component.

- **App-Specific Configurations:**
  - Implement app-specific timeout settings.
  - Develop scheduler strategies that consider application-specific requirements.

## Markov Model Implementation

- **3D State Representation:**
  - The Markov folder contains a continuous-time Markov chain (CTMC) model in model_3D.py.
  - States are represented as 3D tuples (i, j, k) where:
    - i: Number of waiting requests.
    - j: Number of available containers (idle or active).
    - k: Number of requests being processed.
  
- **Transition Rates:**
  - **λ (lambda):** Request arrival rate.
  - **μ (mu):** Service completion rate.
  - **α (alpha):** Container spawning rate.
  - **β (beta):** Request assignment rate.
  - **θ (theta):** Container timeout rate.

- **Performance Metrics:**
  - Blocking ratios: Percentage of requests that are blocked.
  - Waiting requests: Average number of requests waiting in the queue.
  - Processing requests: Average number of requests being processed.
  - Effective arrival rates: Actual arrival rate considering blocked requests.
  - Waiting times: Average time a request waits before being processed.

## Model Comparison Framework

- **Comparative Analysis:**
  - The model_comparison.py file provides a framework to compare the analytical Markov model with simulation results.
  - Parameters are unified between both approaches to ensure fair comparison.
  
- **Configurable Scenarios:**
  - Multiple test scenarios can be defined with different parameter sets:
    - Arrival and service rates.
    - Container spawn and timeout rates.
    - Request assignment rates.
    - Queue size limits.
    - Server resource capacities.

- **Metric Evaluation:**
  - Comparison metrics include:
    - Mean Squared Error (MSE): Overall deviation between model predictions and simulation results.
    - Mean Absolute Percentage Error (MAPE): Relative error expressed as a percentage.
    - Individual metric differences: Detailed comparison of each performance metric.

- **Reporting Features:**
  - Results are saved to CSV files for further analysis.
  - Text reports summarize findings for each scenario.
  - Runtime performance tracking for both approaches.
