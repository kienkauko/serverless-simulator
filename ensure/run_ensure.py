"""Run ENSURE scheduler on a per-minute arrival trace.

Repeats the simulation ``REPETITIONS`` times against ``TRACE_NAME`` (from
``arrival_data/ML_tests/``) and writes a combined per-minute log to
``logs/<trace_stem>/<timestamp>_multi_x{REPETITIONS}_ensure.csv``.

Run:  python -m ensure.run_ensure
"""
from __future__ import annotations

import copy
import csv
import itertools
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import simpy

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from variables import config as base_config
from Server import Server
from dynamic_pool.Container import Container
from ensure.Container import EnsureContainer
from ensure.System import EnsureSystem

# ----------------------------- knobs -----------------------------------------
TRACE_NAME = "non_station.csv"     # one of: day_night.csv, non_station.csv
DAYS_TO_RUN = 30
REPETITIONS = 10
BASE_SEED = 42

# FnScale knobs (overlaid on base_config["ensure"])
ENSURE_KNOBS = {
    "active_window": 5.0,         # seconds — R = active in last 5s
    "scale_check_interval": 5.0,  # seconds between buffer top-ups
    "sqrt_staffing_c": 1.0,       # buffer = ceil(c * sqrt(R))
}

SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440

TRACE_CSV = _REPO_ROOT / "arrival_data" / "ML_tests" / TRACE_NAME
LOG_DIR = _REPO_ROOT / "logs" / TRACE_NAME.replace(".csv", "")

LOG_FIELDS = [
    "repetition", "seed", "minute", "time",
    "arrival", "accepted", "blocked", "cold_hit", "reuse",
    "warm_pool_size", "ensure_R", "ensure_buffer_target", "ensure_buffer_active",
    "energy", "ram_area", "cpu_area",
    "mean_latency",
]


# ----------------------------- helpers ---------------------------------------
def load_trace(csv_path: Path, n_minutes: int) -> np.ndarray:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                m = int(row["minute"])
                v = float(row["count"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append((m, v))
    rows.sort(key=lambda x: x[0])
    series = np.asarray([v for _, v in rows], dtype=float)
    if len(series) < n_minutes:
        raise RuntimeError(
            f"Trace {csv_path.name} has only {len(series)} minutes, need {n_minutes}."
        )
    return series[:n_minutes]


def build_config(sim_time: float) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["system"]["warmup_time"] = 0
    cfg["system"]["verbose"] = False
    cfg["system"]["sim_time"] = sim_time
    # idle_timeout defaults from variables.py; keep ``idle_cpu_timeout``.
    cfg.setdefault("ensure", {}).update(ENSURE_KNOBS)
    return cfg


def run_one_repetition(rep_idx: int, seed: int, cfg: dict,
                        trace: np.ndarray, n_minutes: int) -> EnsureSystem:
    random.seed(seed)
    np.random.seed(seed)
    Container.id_counter = itertools.count()

    env = simpy.Environment()
    system = EnsureSystem(
        env, cfg,
        distribution=cfg["distribution"],
        trace_per_minute=trace,
        n_minutes=n_minutes,
        repetition=rep_idx,
        seed=seed,
        verbose=cfg["system"]["verbose"],
    )
    for i in range(cfg["system"]["num_servers"]):
        system.add_server(Server(env, f"Server-{i}", cfg["server"]))

    env.process(system.minute_ticker())
    env.process(system.request_generator())
    env.process(system.fnscale_process())
    env.process(system.warmup_reset_process())

    env.run(until=cfg["system"]["sim_time"] + 1.0)
    return system


def print_run_summary(system: EnsureSystem):
    stats = system.request_stats
    lat = system.latency_stats
    mean_lat = (lat["total_latency"] / lat["count"]) if lat["count"] > 0 else 0.0
    bp = (stats["blocked_no_server_capacity"] / stats["generated"]
          if stats["generated"] > 0 else 0.0)
    cold = stats["container_spawns_initiated"]
    reuse = stats["container_reuses"]
    gen = stats["generated"]
    cs_ratio = (cold / gen) if gen > 0 else 0.0
    reuse_ratio = (reuse / gen) if gen > 0 else 0.0
    print(f"  rep={system.repetition:>2}  seed={system.seed}  "
          f"gen={gen}  proc={stats['processed']}  "
          f"block={bp * 100:.2f}%  cold={cs_ratio * 100:.1f}%  "
          f"reuse={reuse_ratio * 100:.1f}%  "
          f"mean_lat={mean_lat:.3f}s  mean_power={system.get_mean_power_usage():.1f}W")


def write_log(all_rows, n_reps: int) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    csv_path = LOG_DIR / f"{ts}_multi_x{n_reps}_ensure.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return csv_path


def main():
    n_minutes = DAYS_TO_RUN * MINUTES_PER_DAY
    trace = load_trace(TRACE_CSV, n_minutes)
    sim_time = n_minutes * SECONDS_PER_MINUTE
    cfg = build_config(sim_time)

    print("\n=== ensure/run_ensure.py — ENSURE scheduler, trace-driven multi-run ===")
    print(f"  TRACE_NAME:       {TRACE_NAME}")
    print(f"  DAYS_TO_RUN:      {DAYS_TO_RUN} ({n_minutes} minutes)")
    print(f"  REPETITIONS:      {REPETITIONS}  (BASE_SEED={BASE_SEED})")
    print(f"  sim_time:         {sim_time}s")
    print(f"  Trace mean/min/max: {trace.mean():.2f} / {trace.min():.0f} / {trace.max():.0f}")
    print(f"  idle_timeout:     {cfg['container'].get('idle_cpu_timeout')}s")
    print(f"  FnScale: window={cfg['ensure']['active_window']}s, "
          f"interval={cfg['ensure']['scale_check_interval']}s, "
          f"c={cfg['ensure']['sqrt_staffing_c']}")
    print(f"  Service dist:     {cfg['distribution']['service-distribution']}")
    print(f"  Spawn dist:       {cfg['distribution']['spawn-distribution']}")
    print(f"  Output dir:       {LOG_DIR}")

    all_rows = []
    print("\n--- Repetitions ---")
    for rep in range(REPETITIONS):
        seed = BASE_SEED + rep
        system = run_one_repetition(rep, seed, cfg, trace, n_minutes)
        all_rows.extend(system.minute_log)
        print_run_summary(system)

    csv_path = write_log(all_rows, REPETITIONS)
    print(f"\nCombined per-minute log ({len(all_rows)} rows) written to:")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
