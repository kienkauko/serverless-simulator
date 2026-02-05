# Markov Model for Serverless Deployment

This repository contains scripts to generate a 2D Markov Chain that models the operation of a fixed warm-pool serverless system. From the model, various performance and consumption metrics can be derived. This repository also includes a discrete-event simulator that simulates serverless deployment in response to homogeneous requests arriving at a cluster of homogeneous servers. The simulator is used to verify and validate the Markov model.

> **Note:** A more comprehensive simulator with sophisticated mapping and routing strategies is available in the `main` branch.

---

## Table of Contents
- [Markov Model for Serverless Deployment](#markov-model-for-serverless-deployment)
  - [Table of Contents](#table-of-contents)
  - [How to Run](#how-to-run)
    - [0. Requirements](#0-requirements)
    - [1. Standalone Mode](#1-standalone-mode)
      - [Markov Model](#markov-model)
      - [Simulator](#simulator)
    - [2. Comparison Mode](#2-comparison-mode)
  - [Detailed Simulation Report](#detailed-simulation-report)
  - [Simulator Functions \& Mechanisms](#simulator-functions--mechanisms)
    - [Request Generation](#request-generation)
    - [Container Assignment Mechanism](#container-assignment-mechanism)
    - [Statistics \& Logging](#statistics--logging)
  - [Markov Model Implementation](#markov-model-implementation)
    - [2D State Representation](#2d-state-representation)
    - [Transition Rates](#transition-rates)
    - [Performance Metrics](#performance-metrics)
    - [Resource Metrics](#resource-metrics)
  - [Model Comparison Framework](#model-comparison-framework)
    - [Comparative Analysis](#comparative-analysis)
    - [Configurable Scenarios](#configurable-scenarios)
    - [Metric Evaluation](#metric-evaluation)

---

## How to Run

### 0. Requirements

```bash
pip install -f requirements.txt
```

### 1. Standalone Mode

#### Markov Model
Run the Markov model independently:
```bash
python Markov/model_2D.py
```
- Configuration: the model reads parameters from the `config` object in the `__main__` section (edit there before running).
- Outputs: prints the analytical performance metrics to the console (blocking ratio, throughput, latencies, resource metrics) and returns them from `__main__`.
- Visualization: to draw the 2D Markov graph, uncomment `draw_graph_updated()` in `__main__`. Warning: disable this for large `queue_warm` or `queue_cold` values — plotting very large state spaces can freeze your PC.
 
#### Simulator
Run the discrete-event simulator:
```bash
python main.py
```
- In standalone mode, the simulator reads input parameters from [`variables.py`](variables.py)
- Outputs simulation statistics and performance metrics

### 2. Comparison Mode
Run both the Markov model and simulator with identical parameters:
```bash
python compare_model_simulator.py
```
- Results are compared and stored in the [`/comparison_results`](/comparison_results) folder
- Input metrics are configured in the main section of [`compare_model_simulator.py`](compare_model_simulator.py)
- **Note:** [`variables.py`](variables.py) is not used in this mode

---

## Detailed Simulation Report

**Report Generated:** 10:43 AM on April 16, 2025  
**Last Updated:** February 5, 2026

---

## Simulator Functions & Mechanisms

### Request Generation
- **Arrival Distributions:** Requests can be generated following:
  - **Exponential** (Poisson arrivals)
  - **Weibull** (configurable shape/scale)
  - **Deterministic** (fixed inter-arrival time)
- **Configuration:** Arrival patterns are set in [`variables.py`](variables.py)
- **Resource Demand:** Each request has fixed CPU and RAM requirements


**Cold-Start Distributions:**
- **Exponential:** Memoryless cold-start times
- **Lognormal:** Right-skewed cold-start times (more realistic)
- **Deterministic:** Fixed cold-start duration

Configure in [`variables.py`](variables.py) under `spawn-distribution`.

### Container Assignment Mechanism

The simulator follows this workflow for each arriving request:

1. **Check Idle Containers:**
   - If an idle container is available → assign immediately
   - Otherwise → proceed to spawning

2. **Spawn New Container:**
   - Check server capacity (CPU and RAM)
   - If sufficient resources → spawn container and assign request
   - If all servers are at capacity → **reject request immediately**

3. **Request Rejection:**
   - This is a **loss system** (no queuing, no retries)
   - Rejected requests are blocked permanently

4. **Container Lifecycle:**
   - **Pre-warmed Pool:** System maintains a fixed number of warm containers
   - **Cold-Started Containers:** Removed immediately after completing their job

> **Note:** A system with graceful timeout for idle containers is available in the comprehensive simulator on the `main` branch.

### Statistics & Logging
- **Tracked Metrics:**
  - Total requests generated/processed/blocked
  - Container spawns (initiated/succeeded/failed)
  - Container reuse rate
  - Latency breakdown (waiting/spawning/processing)
  - Resource utilization (CPU/RAM/Energy)

---

## Markov Model Implementation

### 2D State Representation
The continuous-time Markov chain (CTMC) model is implemented in [`Markov/model_2D.py`](/Markov/model_2D.py).

**State Space:** States are represented as 2D tuples `(i, j)` where:
- **i:** Number of jobs processed by warm containers
- **j:** Number of jobs processed by cold-started containers

### Transition Rates

| Symbol | Description | Unit |
|--------|-------------|------|
| **λ** | Request arrival rate | req/s |
| **μ** | Service completion rate (warm containers) | req/s |
| **α** | Effective completion rate (cold containers) | req/s |

> **Note:** α combines the service rate μ and the spawning rate (1/cold-start time)

### Performance Metrics

The model calculates:

| Metric | Description |
|--------|-------------|
| **Blocking Ratio** | Percentage of requests rejected due to lack of resources |
| **Waiting Requests** | Average number of requests waiting for containers |
| **Processing Requests** | Average number of requests currently being served |
| **Effective Arrival Rate** | Actual throughput considering blocked requests |
| **Mean Waiting Time** | Average time from arrival to service start |

### Resource Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **CPU Consumption** | Expected CPU-time over simulation period | CPU%·s |
| **RAM Consumption** | Expected RAM-time over simulation period | RAM%·s |
| **Energy Consumption** | Expected power consumption (Power × time) | Wh |

---

## Model Comparison Framework

### Comparative Analysis
The [`compare_model_simulator.py`](compare_model_simulator.py) script provides a unified framework to validate the analytical Markov model against simulation results.


### Configurable Scenarios
Define multiple test cases with varying parameters:

```python
scenarios = [
    {
        "arrival_rate": 50,
        "service_rate": 0.1,
        "spawn_time": 6.05,
        "num_servers": 10,
        # ... other parameters
    },
    # Additional scenarios...
]
```

**Configurable Parameters:**
- Arrival and service rates
- Container spawn and timeout rates
- Server resource capacities (CPU/RAM)
- Warm pool size

### Metric Evaluation

The comparison framework calculates:

| Error Metric | Formula | Description |
|--------------|---------|-------------|
| **MSE** | Mean((Model - Sim)²) | Overall deviation magnitude |
| **MAPE** | Mean(\|Model - Sim\| / Sim × 100%) | Relative error percentage |
| **Individual Differences** | Model - Sim | Per-metric deviation |

**Output Files:**
- **CSV Results:** Saved to [`/comparison_results`](/comparison_results)
  
---