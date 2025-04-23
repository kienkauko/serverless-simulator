# Serverless Simulator

A simulator for modeling and analyzing serverless computing environments.

## How to Run

### 1. Standalone Mode

#### Markov Model
- Run `/Markov/model_3D.py` to execute the Markov model
- The graph illustration of the Markov state machine can be enabled by uncommenting the `draw_graph_new` function in the main section
- Example of graph output: `/Markov/graph_example.png`

#### Simulator
- Run `main.py` to execute the simulator
- When running in standalone mode, the simulator will take input parameters from `variables.py`

### 2. Comparison Mode
- Run `model_comparison.py` to execute both the Markov model and simulator with the same input parameters
- Results are compared against each other and stored in the `/comparison_results` folder
- Input metrics for this mode can be configured in the main section of `model_comparison.py`
- Note that `variables.py` is not used as input for the simulator in this mode

# Detailed Simulation Report

- **Report Generated:** 10:43AM on 16/04/2025
- **Updated:** 21/04/2025

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

- **Container Assignment Mechanism:**
  - On request arrival, the simulator first checks for any idle container:
    - If an idle container is found, it is immediately assigned to handle the request.
    - Otherwise, a new container is spawned on a server with sufficient capacity.
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
