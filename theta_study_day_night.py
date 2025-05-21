import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Import simulator components
from System import System
from Server import Server
from Request import Request
from Container import Container
from variables import request_stats, latency_stats

# Create directory for saving plots if it doesn't exist
plots_dir = "theta_evaluation_plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Custom configuration instead of using variables.py
custom_config = {
    # System Parameters
    "system": {
        "num_servers": 10,
        "sim_time": 500,  # Simulation time
        "distribution": "deterministic",  # Using deterministic distribution
        "verbose": False,
    },
    
    # Server Parameters
    "server": {
        "cpu_capacity": 100,
        "ram_capacity": 100,
    },
    
    # Request Parameters
    "request": {
        "arrival_rate": 3.0,  # Base arrival rate (will be modulated by day-night pattern)
        "service_rate": 1.0,   # Service rate (1/mean_service_time)
        "cpu_warm": 20,        # CPU resources needed for warm container
        "ram_warm": 20,        # RAM resources needed for warm container
        "cpu_demand": 30,      # CPU resources needed for active container
        "ram_demand": 30,      # RAM resources needed for active container
    },
    
    # Container Parameters
    "container": {
        "spawn_time": 10.0,    # Time to spawn a container
        "idle_timeout": 0,   # Time before an idle container is removed
    }
}

def analyze_time_window(system, start_time, end_time):
    print(f"\nAnalyzing time window: {start_time:.1f} to {end_time:.1f}")
    """
    Analyze metrics for a specific time window
    """
    metrics = {
        "total_requests": 0,
        "blocked_requests": 0,
        "accepted_requests": 0,
        "total_waiting_time": 0.0,
        "avg_waiting_time": 0.0,
        "avg_cpu_usage": 0.0,
        "avg_ram_usage": 0.0
    }
    
    # Analyze requests within the time window
    for req_info in system.request_records:
        # print(req_info)
        # Check if request arrived within the time window
        if start_time <= req_info['arrival_time'] < end_time:
            print(f"Request {req_info}")
            metrics["total_requests"] += 1
            
            if req_info['status'] == "Rejected":
                metrics["blocked_requests"] += 1
            else:
                metrics["accepted_requests"] += 1
                metrics["total_waiting_time"] += req_info['waiting_time']
    # Calculate blocking probability
    metrics["blocking_prob"] = 0.0
    if metrics["total_requests"] > 0:
        metrics["blocking_prob"] = metrics["blocked_requests"] / metrics["total_requests"]
    
    # Calculate average waiting time for accepted requests
    metrics["avg_waiting_time"] = 0.0
    if metrics["accepted_requests"] > 0:
        metrics["avg_waiting_time"] = metrics["total_waiting_time"] / metrics["accepted_requests"]        
        print(f"Total waiting time: {metrics['total_waiting_time']}, "
              f"Accepted requests: {metrics['accepted_requests']}, "
                f"Avg waiting time: {metrics['avg_waiting_time']}")
    # Calculate average resource usage from resource snapshots
    cpu_usage_samples = []
    ram_usage_samples = []
    
    for snapshot in system.resource_history:
        if start_time <= snapshot['time'] < end_time:
            cpu_usage_samples.append(snapshot['cpu_usage_percent'])
            ram_usage_samples.append(snapshot['ram_usage_percent'])
    
    metrics["avg_cpu_usage"] = np.mean(cpu_usage_samples) if cpu_usage_samples else 0.0
    metrics["avg_ram_usage"] = np.mean(ram_usage_samples) if ram_usage_samples else 0.0
    
    return metrics

def run_simulation(config, window_size=10):
    """Run a simulation with the provided configuration."""
    # Reset statistics
    for key in request_stats:
        request_stats[key] = 0
    
    for key in latency_stats:
        latency_stats[key] = 0.0
    
    # Initialize environment
    env = simpy.Environment()
    
    # Create system with day-night traffic pattern
    system = System(env, config, 
                   distribution=config["system"]["distribution"],
                   pattern_type='up_down',  # Use day-night traffic pattern
                   verbose=False)  # Set verbose mode as needed
    
    # Add servers
    for i in range(config["system"]["num_servers"]):
        server = Server(env, f"Server-{i}", 
                       config["server"]["cpu_capacity"], 
                       config["server"]["ram_capacity"])
        system.add_server(server)
    
    # Start the request generation process
    env.process(system.request_generator())
    
    # Start the resource monitoring process to collect CPU and RAM usage data
    env.process(system.resource_monitor_process())
    
    # Track metrics over time
    metrics_over_time = {
        "time": [],
        "arrival_count": [],
        "blocking_prob": [],
        "waiting_time": [],
        "cpu_usage": [],
        "ram_usage": []
    }
    
    # Run the simulation
    print(f"\nStarting simulation with day-night pattern:")
    print(f"- Base arrival rate: {config['request']['arrival_rate']} req/s")
    print(f"- Service rate: {config['request']['service_rate']} req/s")
    print(f"- Number of servers: {config['system']['num_servers']}")
    print(f"- Simulation time: {config['system']['sim_time']} time units")
    print(f"- Container spawn time: {config['container']['spawn_time']} time units")
    print(f"- Container idle timeout: {config['container']['idle_timeout']} time units")
    print(f"- Analysis window size: {window_size} time units\n")
    
    # Run until the configured simulation time
    env.run(until=config["system"]["sim_time"])
    
    # After simulation completes, collect metrics for each time window
    def collect_metrics(system, window_size):
        sim_time = system.env.now
        metrics = {
            "time": [],
            "arrival_count": [],
            "blocking_prob": [],
            "waiting_time": [],
            "cpu_usage": [],
            "ram_usage": []
        }
        
        # Iterate through time windows
        for window_end in range(window_size, int(sim_time) + 1, window_size):
            window_start = window_end - window_size
            
            # Analyze the time window
            window_metrics = analyze_time_window(system, window_start, window_end)
            
            # Record metrics for this window
            metrics["time"].append(window_end)
            metrics["arrival_count"].append(window_metrics["total_requests"])
            metrics["blocking_prob"].append(window_metrics["blocking_prob"])
            metrics["waiting_time"].append(window_metrics["avg_waiting_time"])
            metrics["cpu_usage"].append(window_metrics["avg_cpu_usage"])
            metrics["ram_usage"].append(window_metrics["avg_ram_usage"])
            
            print(f"Time window [{window_start:.1f}-{window_end:.1f}]: "
                  f"Requests: {window_metrics['total_requests']}, "
                  f"Blocking Prob: {window_metrics['blocking_prob']:.4f}, "
                  f"Avg Wait: {window_metrics['avg_waiting_time']:.2f}, "
                  f"CPU: {window_metrics['avg_cpu_usage']:.2f}%, "
                  f"RAM: {window_metrics['avg_ram_usage']:.2f}%")
        
        return metrics
    
    # Collect metrics after simulation has finished
    metrics_over_time = collect_metrics(system, window_size)
    
    # Print summary statistics
    print("\n--- Simulation Statistics ---")
    print(f"Requests Generated: {request_stats['generated']}")
    print(f"Requests Processed: {request_stats['processed']}")
    print(f"Requests Blocked: {request_stats['blocked_no_server_capacity']}")
    
    # Calculate blocking probability
    if request_stats['generated'] > 0:
        blocking_probability = request_stats['blocked_no_server_capacity'] / request_stats['generated']
        print(f"Overall Blocking Probability: {blocking_probability:.4f} ({blocking_probability*100:.2f}%)")
    
    # Calculate average waiting time
    if latency_stats['count'] > 0:
        avg_wait = latency_stats['waiting_time'] / latency_stats['count']
        print(f"Overall Average Waiting Time: {avg_wait:.2f} time units")
    
    # Return metrics collected during simulation
    return metrics_over_time

def plot_metrics(metrics, title_prefix="Day-Night Traffic Study"):
    """Create plots for the collected metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Common time axis for all plots
    time = metrics["time"]
    
    # Figure 1: Traffic Pattern (Request Arrivals)
    plt.figure(figsize=(10, 6))
    plt.plot(time, metrics["arrival_count"], 'b-', linewidth=2)
    plt.title(f"{title_prefix} - Request Arrival Pattern", fontsize=14)
    plt.xlabel("Simulation Time", fontsize=12)
    plt.ylabel("Request Arrivals per Time Window", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{plots_dir}/traffic_pattern_{timestamp}.png")
    
    # Figure 2: Blocking Probability
    plt.figure(figsize=(10, 6))
    plt.plot(time, metrics["blocking_prob"], 'r-', linewidth=2)
    plt.title(f"{title_prefix} - Blocking Probability", fontsize=14)
    plt.xlabel("Simulation Time", fontsize=12)
    plt.ylabel("Blocking Probability", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{plots_dir}/blocking_probability_{timestamp}.png")
    plt.show()
    
    # Figure 3: Average Waiting Time
    plt.figure(figsize=(10, 6))
    plt.plot(time, metrics["waiting_time"], 'g-', linewidth=2)
    plt.title(f"{title_prefix} - Average Waiting Time", fontsize=14)
    plt.xlabel("Simulation Time", fontsize=12)
    plt.ylabel("Average Waiting Time (time units)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{plots_dir}/waiting_time_{timestamp}.png")
    plt.show()
    
    # # Figure 4: CPU Usage
    # plt.figure(figsize=(10, 6))
    # plt.plot(time, metrics["cpu_usage"], 'c-', linewidth=2)
    # plt.title(f"{title_prefix} - CPU Usage", fontsize=14)
    # plt.xlabel("Simulation Time", fontsize=12)
    # plt.ylabel("CPU Usage (%)", fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # # plt.savefig(f"{plots_dir}/cpu_usage_{timestamp}.png")
    # plt.show()
   
    # Figure 5: RAM Usage
    plt.figure(figsize=(10, 6))
    plt.plot(time, metrics["ram_usage"], 'm-', linewidth=2)
    plt.title(f"{title_prefix} - RAM Usage", fontsize=14)
    plt.xlabel("Simulation Time", fontsize=12)
    plt.ylabel("RAM Usage (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{plots_dir}/ram_usage_{timestamp}.png")
    plt.show()
   
    # Figure 6: Combined plot showing all metrics (normalized for comparison)
    plt.figure(figsize=(12, 8))
    
    # Normalize each metric to [0,1] for better comparison
    def normalize(data):
        min_val = min(data) if data and min(data) != max(data) else 0
        max_val = max(data) if data and min(data) != max(data) else 1
        return [(x - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for x in data]
    
    # Plot normalized metrics
    plt.plot(time, normalize(metrics["arrival_count"]), 'b-', linewidth=2, label="Request Arrivals")
    plt.plot(time, normalize(metrics["blocking_prob"]), 'r-', linewidth=2, label="Blocking Probability")
    plt.plot(time, normalize(metrics["waiting_time"]), 'g-', linewidth=2, label="Waiting Time")
    plt.plot(time, normalize(metrics["cpu_usage"]), 'c-', linewidth=2, label="CPU Usage")
    plt.plot(time, normalize(metrics["ram_usage"]), 'm-', linewidth=2, label="RAM Usage")
    
    plt.title(f"{title_prefix} - Combined Normalized Metrics", fontsize=14)
    plt.xlabel("Simulation Time", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{plots_dir}/combined_metrics_{timestamp}.png")
    
    print(f"\nPlots saved to {plots_dir}/ directory with timestamp {timestamp}")

# Entry point
if __name__ == "__main__":
    # Run the simulation with our custom configuration and 10-unit time windows
    simulation_metrics = run_simulation(custom_config, window_size=1)
    
    # Plot the results
    plot_metrics(simulation_metrics)
    
    # Show plots (uncomment if running in interactive environment)
    plt.show()