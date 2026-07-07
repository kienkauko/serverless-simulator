#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of the 3D dynamic-pool CTMC (Markov/model_3D.py) against the
event-driven dynamic-pool simulator (dynamic_pool/) — multiprocessing version.

The CTMC state is s = (x1, x2, x3) = (cold-starting, active, warm-idle)
containers, with transitions:
    arrival          rate lambda   (warm hit | cold start | block)
    spawn complete   rate x1*alpha (alpha = spawn_rate)
    service complete rate x2*mu    (mu = service_rate)
    warm timeout     rate x3*theta (theta = 1 / idle_timeout)

The simulator's distribution laws are chosen by the knobs below
(ARRIVAL_DIST / SERVICE_DIST / SPAWN_DIST / IDLE_DIST). The CTMC always assumes
the memoryless (Markov) versions, so the closest agreement is expected with all
four set to exponential / "markov". Setting IDLE_DIST="deterministic" (a fixed
idle timer) lets you measure how far the model drifts from a deterministically
timed-out warm pool — the idle law is honoured natively by the simulator via the
``idle-distribution`` config field (no monkeypatching).

Each random test case is solved by both models and the per-metric MAPE / RMSE /
R^2 (with bootstrap CIs across cases) are reported.

Run from the repo root:  python validation/3D/CTMC_vs_simulator_multi_core.py
"""

import contextlib
import copy
import io
import math
import os
import random
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import simpy
import warnings
warnings.filterwarnings('ignore')

# --- make the repo root importable regardless of where we are launched from ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Markov model under test
from Markov.model_3D import MarkovModel

# Simulator components (dynamic / idle-timeout pool)
from variables import config as base_config
from Server import Server
from dynamic_pool.System import System
from dynamic_pool.Container import Container

PEAK_POWER = 150.0

# Simulation length (time units). Long enough for the warm-pool occupancy to
# reach steady state; the leading WARMUP_TIME is discarded.
WARMUP_TIME = 1000.0
SIM_TIME = 10000.0

# CTMC tractability: the 3D state space has ~ (n+1)(n+2)(n+3)/6 states, so the
# total number of container slots n is kept small for the dense solve.
MAX_SLOTS = 20

# Simulator distribution laws. The CTMC always assumes the memoryless versions,
# so all-exponential / "markov" gives the closest match. Set IDLE_DIST to
# "deterministic" to validate against a fixed idle timer instead.
ARRIVAL_DIST = "exponential"   # "exponential" | "deterministic" | "weibull"
SERVICE_DIST = "exponential"   # "exponential" | "deterministic" | "traces"
SPAWN_DIST = "exponential"     # "exponential" | "deterministic" | "lognormal" | "trace"
IDLE_DIST = "deterministic"           # "markov" (exponential) | "deterministic"

# Comparison / CI configuration
CI_LEVEL = 0.90
BOOTSTRAP_ITERATIONS = 10000

METRICS = ['blocking_probability', 'latency', 'cpu_usage', 'ram_usage', 'power_usage']


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(data, stat_func=np.mean, n_boot=BOOTSTRAP_ITERATIONS,
                 ci_level=CI_LEVEL, rng=None):
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        val = float(data[0])
        return (val, val, val)
    if rng is None:
        rng = np.random.default_rng()
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([stat_func(data[idx]) for idx in boot_indices])
    alpha = 1 - ci_level
    ci_low = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    point = float(stat_func(data))
    return (ci_low, ci_high, point)


# ---------------------------------------------------------------------------
# Test-case generation
# ---------------------------------------------------------------------------
def generate_test_cases(num_cases=150, seed=42):
    """IID test cases by independent uniform sampling.

    Demands are drawn high enough (>= 20%) that each server hosts only a few
    containers, keeping the total slot count n = containers_per_server *
    num_servers within MAX_SLOTS so the dense 3D CTMC stays solvable. Active
    (demand) resources always strictly exceed both warm and transit resources.
    """
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < num_cases:
        cpu_warm = int(rng.integers(0, 6))
        ram_warm = int(rng.integers(0, 6))
        cpu_transit = int(rng.integers(0, 41))
        ram_transit = int(rng.integers(0, 41))

        # Active demand must exceed both warm and transit; >= 20 caps slots/server.
        cpu_demand_lo = max(10, max(cpu_warm, cpu_transit) + 1)
        ram_demand_lo = max(10, max(ram_warm, ram_transit) + 1)
        if cpu_demand_lo > 100 or ram_demand_lo > 100:
            continue
        cpu_demand = int(rng.integers(cpu_demand_lo, 101))
        ram_demand = int(rng.integers(ram_demand_lo, 101))

        max_resource = max(cpu_demand, ram_demand)
        containers_per_server = math.floor(100 / max_resource)
        if containers_per_server < 1:
            continue

        num_servers = int(rng.integers(1, 7))
        n_slots = containers_per_server * num_servers
        if n_slots < 2 or n_slots > MAX_SLOTS:
            continue

        mean_service_time = float(rng.uniform(0.1, 20.0))
        spawn_time = float(rng.uniform(1.0, 30.0))
        idle_timeout = float(rng.uniform(1.0, 100.0))
        service_rate = 1.0 / mean_service_time

        capacity = n_slots * service_rate
        load_frac = float(rng.uniform(0.3, 1.2))
        arrival_rate = load_frac * capacity

        cases.append({
            'case_id': len(cases),
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'spawn_time': spawn_time,
            'idle_timeout': idle_timeout,
            'num_servers': num_servers,
            'n_slots': n_slots,
            'peak_power': PEAK_POWER,
            'power_scale': float(rng.uniform(0.2, 0.6)),
            'cpu_warm': cpu_warm,
            'ram_warm': ram_warm,
            'cpu_transit': cpu_transit,
            'ram_transit': ram_transit,
            'cpu_demand': cpu_demand,
            'ram_demand': ram_demand,
        })
    return cases


# ---------------------------------------------------------------------------
# Config conversion
# ---------------------------------------------------------------------------
def convert_to_markov_config(tc):
    return {
        "arrival_rate": tc['arrival_rate'],
        "service_rate": tc['service_rate'],
        "spawn_rate": 1.0 / tc['spawn_time'],
        "theta": 1.0 / tc['idle_timeout'],
        "max_queue": tc['n_slots'],
        "ram_warm": tc['ram_warm'],
        "cpu_warm": tc['cpu_warm'],
        "ram_demand": tc['ram_demand'],
        "cpu_demand": tc['cpu_demand'],
        "ram_transit": tc['ram_transit'],
        "cpu_transit": tc['cpu_transit'],
        "power_max": tc['peak_power'],
        "power_min_scale": tc['power_scale'],
    }


def convert_to_simulator_config(tc):
    cfg = copy.deepcopy(base_config)

    cfg["system"]["num_servers"] = tc['num_servers']
    cfg["system"]["warmup_time"] = WARMUP_TIME
    cfg["system"]["sim_time"] = SIM_TIME
    cfg["system"]["verbose"] = False
    cfg["system"]["warm_percent"] = 0

    cfg["request"]["arrival_rate_mean"] = tc['arrival_rate']
    cfg["request"]["arrival_rate_std"] = 0
    cfg["request"]["service_rate"] = tc['service_rate']
    cfg["request"]["warm_cpu"] = tc['cpu_warm']
    cfg["request"]["warm_ram"] = tc['ram_warm']
    cfg["request"]["cold_start_cpu"] = tc['cpu_transit']
    cfg["request"]["cold_start_ram"] = tc['ram_transit']
    cfg["request"]["cpu_demand"] = tc['cpu_demand']
    cfg["request"]["ram_demand"] = tc['ram_demand']

    cfg["container"]["spawn_time_mean"] = tc['spawn_time']
    cfg["container"]["spawn_time_std"] = 0
    cfg["container"]["idle_timeout"] = tc['idle_timeout']
    cfg["container"]["idle_cpu_timeout"] = tc['idle_timeout']

    cfg["server"]["peak_power"] = tc['peak_power']
    cfg["server"]["power_scale"] = tc['power_scale']

    # Distribution laws chosen by the user knobs above.
    cfg["distribution"]["arrival-distribution"] = ARRIVAL_DIST
    cfg["distribution"]["service-distribution"] = SERVICE_DIST
    cfg["distribution"]["spawn-distribution"] = SPAWN_DIST
    cfg["distribution"]["idle-distribution"] = IDLE_DIST

    return cfg


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_markov_model(markov_config):
    try:
        model = MarkovModel(markov_config, verbose=False)
        m = model.get_metrics()
        return {
            'blocking_probability': m['blocking_ratios'][0],
            'latency': m['latency'][0],
            'cpu_usage': m['cpu_usage'][0],
            'ram_usage': m['ram_usage'][0],
            'power_usage': m['power_usage'][0],
        }
    except Exception as e:
        print(f"Error in Markov model: {e}")
        return None


def run_simulator(sim_config, seed):
    try:
        random.seed(seed)
        np.random.seed(seed)
        Container.id_counter = __import__('itertools').count()

        with contextlib.redirect_stdout(io.StringIO()):
            env = simpy.Environment()
            system = System(env, sim_config,
                            distribution=sim_config["distribution"],
                            verbose=False)
            for i in range(sim_config["system"]["num_servers"]):
                system.add_server(Server(env, f"Server-{i}", sim_config["server"]))

            env.process(system.request_generator())
            env.process(system.warmup_reset_process())
            env.run(until=sim_config["system"]["sim_time"])

        stats = system.request_stats
        generated = stats['generated']
        blocked = stats['blocked_no_server_capacity']
        count = system.latency_stats['count']

        return {
            'blocking_probability': (blocked / generated) if generated > 0 else 0.0,
            'latency': (system.latency_stats['total_latency'] / count) if count > 0 else 0.0,
            'cpu_usage': system.get_mean_cpu_usage(),
            'ram_usage': system.get_mean_ram_usage(),
            'power_usage': system.get_mean_power_usage(),
        }
    except Exception as e:
        print(f"Error in simulator: {e}")
        return None


def run_single_case(tc):
    markov_config = convert_to_markov_config(tc)
    sim_config = convert_to_simulator_config(tc)
    markov_result = run_markov_model(markov_config)
    sim_result = run_simulator(sim_config, seed=42 + tc['case_id'])
    return {
        'test_case': tc,
        'markov_config': markov_config,
        'markov_result': markov_result,
        'sim_result': sim_result,
    }


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def calculate_comparison_metrics(markov_results, sim_results):
    comparison = {}
    for metric in METRICS:
        markov_vals = np.array([r[metric] for r in markov_results if r is not None])
        sim_vals = np.array([r[metric] for r in sim_results if r is not None])
        n = min(len(markov_vals), len(sim_vals))
        markov_vals, sim_vals = markov_vals[:n], sim_vals[:n]
        if n == 0:
            comparison[metric] = {'MAPE': float('inf'), 'RMSE': float('inf'),
                                  'R_squared': 0.0, 'n_samples': 0}
            continue

        errors = markov_vals - sim_vals
        denom = np.where(np.abs(sim_vals) > 1e-6, sim_vals, 1.0)
        per_case_ape = np.abs(errors / denom) * 100
        mape = float(np.mean(per_case_ape))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((sim_vals - np.mean(sim_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        boot_rng = np.random.default_rng(42)
        if n > 1:
            ci_mape_low, ci_mape_high, _ = bootstrap_ci(
                per_case_ape, stat_func=np.mean, ci_level=CI_LEVEL, rng=boot_rng)
            ci_sim_low, ci_sim_high, sim_mean = bootstrap_ci(
                sim_vals, stat_func=np.mean, ci_level=CI_LEVEL, rng=boot_rng)
        else:
            ci_mape_low = ci_mape_high = mape
            sim_mean = float(sim_vals[0])
            ci_sim_low = ci_sim_high = sim_mean

        comparison[metric] = {
            'MAPE': mape, 'RMSE': rmse, 'R_squared': r_squared, 'n_samples': n,
            'ci_mape_low': ci_mape_low, 'ci_mape_high': ci_mape_high,
            'sim_mean': sim_mean, 'ci_sim_low': ci_sim_low, 'ci_sim_high': ci_sim_high,
            'markov_mean': float(np.mean(markov_vals)),
        }
    return comparison


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("3D dynamic-pool CTMC vs Simulator validation (multi-core)")
    print("=" * 60)

    num_workers = 4
    num_cases = 500
    print(f"Workers: {num_workers}   Cases: {num_cases}")
    print(f"Sim time: {SIM_TIME} (warmup {WARMUP_TIME}), max slots: {MAX_SLOTS}")
    print(f"Distributions: arrival={ARRIVAL_DIST}, service={SERVICE_DIST}, "
          f"spawn={SPAWN_DIST}, idle={IDLE_DIST}")

    test_cases = generate_test_cases(num_cases)
    print(f"Generated {len(test_cases)} test cases\n")

    with Pool(processes=num_workers) as pool:
        results = pool.map(run_single_case, test_cases)

    markov_results, sim_results, detailed = [], [], []
    for outcome in results:
        tc = outcome['test_case']
        mk = outcome['markov_result']
        sm = outcome['sim_result']
        if mk is None or sm is None:
            print(f"  Case {tc['case_id']:>3d}: FAILED")
            continue
        markov_results.append(mk)
        sim_results.append(sm)

        row = {
            'case_id': tc['case_id'],
            'arrival_rate': tc['arrival_rate'],
            'service_rate': tc['service_rate'],
            'spawn_time': tc['spawn_time'],
            'idle_timeout': tc['idle_timeout'],
            'num_servers': tc['num_servers'],
            'n_slots': tc['n_slots'],
            'cpu_warm': tc['cpu_warm'], 'ram_warm': tc['ram_warm'],
            'cpu_transit': tc['cpu_transit'], 'ram_transit': tc['ram_transit'],
            'cpu_demand': tc['cpu_demand'], 'ram_demand': tc['ram_demand'],
        }
        aliases = {'blocking': 'blocking_probability', 'latency': 'latency',
                   'cpu': 'cpu_usage', 'ram': 'ram_usage', 'power': 'power_usage'}
        for short, full in aliases.items():
            row[f'markov_{short}'] = mk[full]
            row[f'sim_{short}'] = sm[full]
        detailed.append(row)

        print(f"  Case {tc['case_id']:>3d} (n={tc['n_slots']:>2d}): "
              f"block(M={mk['blocking_probability']:.4f}, S={sm['blocking_probability']:.4f})  "
              f"lat(M={mk['latency']:.3f}, S={sm['latency']:.3f})")

    print(f"\nComparing {len(markov_results)} successful cases...")
    comparison = calculate_comparison_metrics(markov_results, sim_results)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    ci_pct = int(CI_LEVEL * 100)
    for metric in METRICS:
        m = comparison[metric]
        print(f"\n  {metric.upper().replace('_', ' ')}  (n={m['n_samples']}):")
        print(f"    Sim mean:    {m['sim_mean']:.6f}   "
              f"{ci_pct}% CI: [{m['ci_sim_low']:.6f}, {m['ci_sim_high']:.6f}]")
        print(f"    Markov mean: {m['markov_mean']:.6f}")
        print(f"    MAPE:        {m['MAPE']:.2f}%   "
              f"{ci_pct}% CI: [{m['ci_mape_low']:.2f}%, {m['ci_mape_high']:.2f}%]")
        print(f"    RMSE:        {m['RMSE']:.6f}")
        print(f"    R-squared:   {m['R_squared']:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_HERE, "comparison_results")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"markov3D_vs_sim_{ts}.csv")
    pd.DataFrame(detailed).to_csv(filename, index=False)
    print(f"\nDetailed results saved to: {filename}")
    print(f"Successful comparisons: {len(markov_results)}/{len(test_cases)} "
          f"({len(markov_results)/len(test_cases)*100:.1f}%)")


if __name__ == "__main__":
    main()
