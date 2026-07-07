import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import simpy
import numpy as np

from variables import config
from Server import Server
from fixed_pool.System import System

# --- Simulation Setup ---

env = simpy.Environment()

system = System(env, config, distribution=config["distribution"], verbose=config["system"]["verbose"])

for i in range(config["system"]["num_servers"]):
    server = Server(env, f"Server-{i}", config["server"])
    system.add_server(server)

# Pre-warm containers, then start request generation
pre_warm_done = env.process(system.pre_warm())

def start_request_generator():
    yield pre_warm_done
    env.process(system.request_generator())

env.process(start_request_generator())
env.process(system.warmup_reset_process())

# --- Run ---

print(f"\nStarting simulation:")
print(f"  Arrival Rate (λ):   {config['request']['arrival_rate_mean']} req/s  (σ={config['request']['arrival_rate_std']})")
print(f"  Arrival Distribution: {config['distribution']['arrival-distribution']}")
print(f"  Service Rate (μ):   {config['request']['service_rate']} req/s")
print(f"  Service Distribution: {config['distribution']['service-distribution']}")
print(f"  Servers:            {config['system']['num_servers']}")
print(f"  Simulation Time:    {config['system']['sim_time']}s")
print(f"  Spawn Time Mean:    {config['container']['spawn_time_mean']}s  (σ={config['container']['spawn_time_std']})")
print(f"  Spawn Distribution: {config['distribution']['spawn-distribution']}")
print(f"  Idle Timeout:       {config['container']['idle_cpu_timeout']}s")
print(f"  Warm Pool Size:     {system.max_warm_queue} containers ({config['system']['warm_percent']*100:.0f}%)\n")

env.run(until=config["system"]["sim_time"])

# --- Results ---

print("\n--- Simulation Statistics ---")
print(f"  Requests Generated:             {system.request_stats['generated']}")
print(f"  Requests Processed:             {system.request_stats['processed']}")
print(f"  Requests Blocked (No Capacity): {system.request_stats['blocked_no_server_capacity']}")

if system.request_stats['generated'] > 0:
    bp = system.request_stats['blocked_no_server_capacity'] / system.request_stats['generated']
    print(f"  Blocking Probability:           {bp:.4f} ({bp*100:.2f}%)")

print(f"  Container Spawns Initiated:     {system.request_stats['container_spawns_initiated']}")
print(f"  Container Spawns Succeeded:     {system.request_stats['container_spawns_succeeded']}")

if system.latency_stats['count'] > 0:
    n = system.latency_stats['count']
    avg_total = system.latency_stats['total_latency'] / n
    avg_spawn = system.latency_stats['spawning_time'] / n
    avg_wait  = system.latency_stats['waiting_time'] / n
    avg_proc  = system.latency_stats['processing_time'] / n

    e_br2 = system.latency_stats['total_latency_sq'] / n
    var_lat = e_br2 - avg_total ** 2
    std_lat = var_lat ** 0.5 if var_lat >= 0 else 0.0

    all_lat = system.latency_stats['all_latencies']
    p50 = float(np.percentile(all_lat, 50))
    p95 = float(np.percentile(all_lat, 95))
    p99 = float(np.percentile(all_lat, 99))

    print("\n--- Average Latencies ---")
    print(f"  Total End-to-End: {avg_total:.4f}s")
    print(f"  Spawn Time:       {avg_spawn:.4f}s")
    print(f"  Wait Time:        {avg_wait:.4f}s")
    print(f"  Processing Time:  {avg_proc:.4f}s")
    print(f"  Variance:         {var_lat:.4f}s²")
    print(f"  Std Dev:          {std_lat:.4f}s")
    print(f"  P50 (Median):     {p50:.4f}s")
    print(f"  P95:              {p95:.4f}s")
    print(f"  P99:              {p99:.4f}s")

if system.env.now > 0:
    avg_waiting = system.total_waiting_area / system.env.now
    avg_running = system.get_mean_processing_count()
    avg_in_sys  = system.get_mean_requests_in_system()
    eff_rate    = system.request_stats['processed'] / system.env.now
    ll_wait     = avg_waiting / eff_rate if eff_rate > 0 else 0

    print("\n--- Queue Metrics (Little's Law) ---")
    print(f"  Avg waiting requests:    {avg_waiting:.2f}")
    print(f"  Avg processing requests: {avg_running:.2f}")
    print(f"  Avg requests in system:  {avg_in_sys:.2f}")
    print(f"  Effective arrival rate:  {eff_rate:.2f} req/s")
    print(f"  Predicted wait time:     {ll_wait:.2f}s")

print("\n--- Resource Utilization ---")
print(f"  Mean CPU Usage: {system.get_mean_cpu_usage():.4f}%")
print(f"  Mean RAM Usage: {system.get_mean_ram_usage():.4f}%")
