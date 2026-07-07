"""Multi-run dynamic-pool simulator (RL-driven or static idle-timeout sweep).

This script has two modes, selected by the ``USE_ML`` knob.

USE_ML=True  — RL-driven trace replay
    Arrivals follow per-minute counts in ``arrival_data/ML_tests/<TRACE_NAME>``
    (exponential interarrivals within each minute at rate = count / 60).
    Every ``STEP_DURATION`` simulated seconds the trained SAC/PPO policy is
    queried with [arrivals, warm_pool_size, cold_hits, ram_util] and the
    resulting idle timeout is applied to ``system.idle_timeout`` (affecting
    newly-spawned containers). One combined CSV is written per run.

USE_ML=False — static idle-timeout sweep (RL disabled)
    The arrival trace is ignored. Arrivals are drawn from ``variables.py``
    (``request.arrival_rate_mean`` / ``arrival_rate_std`` via the configured
    arrival distribution) and the run lasts one full ``system.sim_time``.
    The simulation is repeated once per value in ``IDLE_TIMEOUTS``, each with a
    fixed ``idle_cpu_timeout``, so the effect of that setpoint can be studied.
    Because the traffic is stationary Poisson, per-minute detail is not kept:
    each run is collapsed to a single whole-run average and all idle-timeout
    settings are collocated in ONE CSV (``logs/dynamic/``), with a ``timeout``
    column distinguishing the rows.

Per-minute log fields — USE_ML=True (one row per (repetition, minute)):
  - repetition, seed, minute, time
  - arrival, accepted, warm_pool_size
  - cold_hit   (cumulative reactive cold starts since run start; running total)
  - redundant  (cumulative idle-container-seconds since run start: the integral
               of the idle warm-pool pod count over time, in pod*s)
  - p95_active, p99_active
  - idle_timeout (current setpoint at minute start)
  - energy, ram_area, cpu_area, free_ram_time, mean_latency
    (free_ram_time = cumulative warm/idle RAM-seconds: RAM held by warm
    containers while idling; cold-start/processing RAM excluded)
  - p95_latency, p99_latency (whole-run latency percentiles; constant per
    repetition — same value on every minute row of that run)

Summary fields — USE_ML=False (one row per (timeout, repetition)):
  - timeout, repetition, seed, sim_time
  - generated, accepted, blocked, blocking_prob
  - cold_start, cold_start_ratio, reuse_ratio
  - mean_warm_pool, mean_redundant, p95_active, p99_active
  - mean_cold_starting, mean_processing, mean_idle  (time-weighted container-state averages)
  - mean_latency, p95_latency, p99_latency  (whole-run latency stats)
  - mean_cpu, mean_ram, mean_power
  - free_ram_time (whole-run warm/idle RAM-seconds; wasted idle RAM)

Run:  python -m dynamic_pool.RL.infer
"""

from __future__ import annotations

import copy
import csv
import itertools
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import simpy

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from variables import config as base_config
from Server import Server
from Request import Request
from dynamic_pool.System import System
from dynamic_pool.Container import Container

# ----------------------------- knobs -----------------------------------------
# USE_ML / TRACE_NAME / NUM_WORKERS honor env overrides (SIM_USE_ML /
# SIM_TRACE_NAME / SIM_NUM_WORKERS) so run_all_cases.py can drive batches and the
# overrides survive multiprocessing-spawn re-imports; defaults are unchanged.
USE_ML = os.environ.get("SIM_USE_ML", "False").strip().lower() in ("1", "true", "yes")
# True  -> arrivals follow per-minute counts in arrival_data/ML_tests/<TRACE_NAME>.
# False -> arrivals drawn from variables.py (arrival_rate_mean / std via
#          the configured arrival distribution); sim_time is taken from
#          variables.py instead of being derived from DAYS_TO_RUN.
USE_TRACE = True
ALGORITHM = "PPO"         # "PPO" or "SAC" — must match the trained checkpoint
# True  -> load the best checkpoint (highest eval reward seen during training).
# False -> load the final-step policy. Best usually generalizes better.
USE_BEST = True
TRACE_NAME = os.environ.get("SIM_TRACE_NAME", "day_night.csv")   # "non_station.csv" or "day_night.csv"
DAYS_TO_RUN = 30
NUM_REPETITIONS = 10
BASE_SEED = 42
# Number of worker processes used to run repetitions in parallel. None -> auto
# (min(NUM_REPETITIONS, os.cpu_count())). Each repetition runs in its own
# process; all results are gathered (barrier) before any file is written, so
# rows always land in repetition order with no cross-process write races.
NUM_WORKERS = int(os.environ.get("SIM_NUM_WORKERS", "4"))

# --- USE_ML=False only -------------------------------------------------------
# If SWEEP=True, one simulation (of NUM_REPETITIONS runs) is executed per value
# in IDLE_TIMEOUTS, each with a fixed idle_cpu_timeout.
# If SWEEP=False, a single simulation is executed using idle_cpu_timeout from
# variables.py.
SWEEP = False
IDLE_TIMEOUTS = [x / 2.0 for x in range(0, 41)]
STATIC_LOG_DIR = _REPO_ROOT / "logs" / "dynamic"

_TRACE_CONFIG = {
    "non_station.csv": {"log_subdir": "non_station"},
    "day_night.csv":   {"log_subdir": "day_night"},
}

if TRACE_NAME not in _TRACE_CONFIG:
    raise ValueError(
        f"Unknown TRACE_NAME {TRACE_NAME!r}; "
        f"expected one of {sorted(_TRACE_CONFIG)}."
    )

_TRACE_STEM = TRACE_NAME.replace(".csv", "")
TRACE_CSV = _REPO_ROOT / "arrival_data" / "ML_tests" / TRACE_NAME
# Checkpoint paths are algorithm- and trace-derived, matching dynamic_pool/RL/train.py.
_RUN_STEM = f"{ALGORITHM.lower()}_idle_timeout_{_TRACE_STEM}"
if USE_BEST:
    MODEL_PATH = _HERE / f"{_RUN_STEM}_best" / "best_model"
    VECNORM_PATH = _HERE / f"{_RUN_STEM}_best" / "best_vecnorm.pkl"
else:
    MODEL_PATH = _HERE / _RUN_STEM
    VECNORM_PATH = _HERE / f"{_RUN_STEM}_vecnorm.pkl"
LOG_DIR = _REPO_ROOT / "logs" / _TRACE_CONFIG[TRACE_NAME]["log_subdir"]

STEP_DURATION = 60.0      # RL decision interval (s) — also the per-minute tick
IDLE_MIN = 1.0
IDLE_MAX = 60.0

SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440

LOG_FIELDS = [
    "repetition", "seed", "minute", "time",
    "arrival", "accepted", "warm_pool_size",
    "cold_hit", "redundant",
    "p95_active", "p99_active",
    "idle_timeout",
    "energy", "ram_area", "cpu_area", "free_ram_time",
    "mean_latency", "p95_latency", "p99_latency",
]

# Whole-run summary — one row per (idle_cpu_timeout, repetition) for USE_ML=False.
SUMMARY_FIELDS = [
    "timeout", "repetition", "seed", "sim_time",
    "generated", "accepted", "blocked", "blocking_prob",
    "cold_start", "cold_start_ratio", "reuse_ratio",
    "mean_warm_pool", "mean_redundant", "p95_active", "p99_active",
    "mean_cold_starting", "mean_processing", "mean_idle",
    "mean_latency", "p95_latency", "p99_latency",
    "mean_cpu", "mean_ram", "mean_power", "free_ram_time",
]


# Set once per repetition; the patched Container hooks attribute Idle<->Active
# transitions to the currently-active system.
_ACTIVE_TRACKER = None
_CONTAINER_HOOKS_INSTALLED = False


def _install_container_hooks():
    """Wrap dynamic_pool.Container.assign_request / release_request to record
    Idle<->Active transitions in the live system's per-minute histogram."""
    global _CONTAINER_HOOKS_INSTALLED
    if _CONTAINER_HOOKS_INSTALLED:
        return
    _CONTAINER_HOOKS_INSTALLED = True

    orig_assign = Container.assign_request
    orig_release = Container.release_request

    def assign_request(self, request):
        result = orig_assign(self, request)
        if _ACTIVE_TRACKER is not None:
            _ACTIVE_TRACKER._record_active_delta(+1)
        return result

    def release_request(self):
        was_active = self.current_request is not None
        orig_release(self)
        if was_active and _ACTIVE_TRACKER is not None:
            _ACTIVE_TRACKER._record_active_delta(-1)

    Container.assign_request = assign_request
    Container.release_request = release_request


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
        raise RuntimeError(f"Trace has only {len(series)} minutes, need {n_minutes}.")
    return series[:n_minutes]


def build_config(sim_time: float, idle_timeout=None) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["system"]["warmup_time"] = 0
    cfg["system"]["verbose"] = False
    cfg["system"]["sim_time"] = sim_time
    if idle_timeout is not None:
        cfg["container"]["idle_timeout"] = idle_timeout
        cfg["container"]["idle_cpu_timeout"] = idle_timeout
    return cfg


# ----------------------------- RL-aware System -------------------------------
class RLSystem(System):
    """Dynamic-pool System extension with per-minute metric logging.

    Two arrival modes:
      * trace-driven   (``trace_per_minute`` given) — arrivals injected per
        minute from the trace; idle_timeout optionally RL-controlled.
      * synthetic      (``trace_per_minute`` is None) — arrivals come from the
        base exponential/weibull generator (``arrival_rate_mean`` etc.); the
        minute_ticker only ticks for logging.
    """

    def __init__(self, env, config, distribution, trace_per_minute,
                 use_ml, model, obs_normalizer, repetition, seed,
                 n_minutes=None, verbose=False):
        super().__init__(env, config, distribution, verbose=verbose)
        if trace_per_minute is not None:
            self.trace = np.asarray(trace_per_minute, dtype=float)
            self.trace_driven = True
            self.n_minutes = len(self.trace)
        else:
            self.trace = None
            self.trace_driven = False
            if n_minutes is None:
                raise ValueError("n_minutes is required when no trace is provided.")
            self.n_minutes = int(n_minutes)
        self.use_ml = use_ml
        self.model = model
        self.obs_normalizer = obs_normalizer
        self.repetition = repetition
        self.seed = seed

        # Per-minute counters
        self.minute_arrivals_count = 0
        self.minute_cold_start_count = 0
        self.minute_blocked_count = 0
        self.minute_log = []

        # Cumulative (accumulated) metrics — running totals across the run.
        # cumulative_cold_hit: reactive cold starts summed over all minutes.
        # total_redundant_pod_time: idle-container-seconds, i.e. the integral of
        # the idle warm-pool pod count over time. Flushed via the same points as
        # the base idle stats (see update_idle_stats override below); unlike the
        # base total_idle_area it is never reset at warmup, so it stays a pure
        # running total from t=0.
        self.cumulative_cold_hit = 0
        self.total_redundant_pod_time = 0.0

        # Active-container time-weighted histogram (per-minute, reset each tick).
        self._active_count = 0
        self._last_active_t = 0.0
        self._minute_level_time = defaultdict(float)

        # RL observation tracking (running tallies that the base System keeps).
        self._prev_gen = 0
        self._prev_cold = 0
        self._prev_ram_area = 0.0

    # ----------------------- arrival generation ------------------------------
    def request_generator(self):
        """Trace-driven runs emit arrivals from minute_ticker; synthetic runs
        fall back to the base exponential/weibull generator."""
        if not self.trace_driven:
            yield from super().request_generator()
        return

    def minute_ticker(self):
        """One iteration per simulated minute: optionally query RL, inject
        trace arrivals, then accumulate per-minute metrics."""
        obs = np.zeros(4, dtype=np.float32)
        span = IDLE_MAX - IDLE_MIN

        for m in range(self.n_minutes):
            minute_start = self.env.now

            # ------------- RL decision (if enabled) -------------
            if self.use_ml and self.model is not None:
                model_obs = (self.obs_normalizer.normalize_obs(obs)
                             if self.obs_normalizer is not None else obs)
                action, _ = self.model.predict(model_obs, deterministic=True)
                a = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))
                new_timeout = IDLE_MIN + (a + 1.0) * 0.5 * span
                self.idle_timeout = new_timeout

            # ------------- Reset per-minute counters -------------
            self.minute_arrivals_count = 0
            self.minute_cold_start_count = 0
            self.minute_blocked_count = 0

            self._minute_level_time = defaultdict(float)
            self._last_active_t = minute_start

            # ------------- Inject trace arrivals (trace mode only) -------------
            if self.trace_driven:
                expected_rate = max(0.0, float(self.trace[m]))
                if expected_rate > 0:
                    self.env.process(self._inject_arrivals_exponential(expected_rate))

            yield self.env.timeout(SECONDS_PER_MINUTE)

            # ------------- End-of-minute bookkeeping -------------
            # Accumulate cold starts into the running total for this minute.
            self.cumulative_cold_hit += self.minute_cold_start_count

            self.update_resource_stats()
            mean_lat = (self.latency_stats['total_latency'] / self.latency_stats['count']
                        if self.latency_stats['count'] > 0 else 0.0)

            # Flush trailing segment of the active-count histogram.
            now = self.env.now
            trailing = now - self._last_active_t
            if trailing > 0:
                self._minute_level_time[self._active_count] += trailing
            self._last_active_t = now
            p95_active, p99_active = self._percentiles_from_hist(
                self._minute_level_time, SECONDS_PER_MINUTE
            )

            # ------------- Build next RL observation -------------
            gen_now = self.request_stats["generated"]
            cold_now = self.request_stats["container_spawns_initiated"]
            d_gen = gen_now - self._prev_gen
            d_cold = cold_now - self._prev_cold
            self._prev_gen, self._prev_cold = gen_now, cold_now

            ram_area_now = self.total_ram_usage_area
            d_ram_area = ram_area_now - self._prev_ram_area
            self._prev_ram_area = ram_area_now
            ram_util = (d_ram_area / (STEP_DURATION * self.total_ram_capacity)
                        if self.total_ram_capacity > 0 else 0.0)

            obs = np.array(
                [d_gen, len(self.idle_containers), d_cold, ram_util],
                dtype=np.float32,
            )

            self.minute_log.append({
                "repetition": self.repetition,
                "seed": self.seed,
                "minute": m,
                "time": self.env.now,
                "arrival": self.minute_arrivals_count,
                "accepted": self.minute_arrivals_count - self.minute_blocked_count,
                "warm_pool_size": len(self.idle_containers),
                "cold_hit": self.cumulative_cold_hit,
                "redundant": self.get_redundant_pod_time(),
                "p95_active": p95_active,
                "p99_active": p99_active,
                "idle_timeout": self.idle_timeout,
                "energy": self.total_energy_usage,
                "ram_area": self.total_ram_usage_area,
                "cpu_area": self.total_cpu_usage_area,
                "free_ram_time": self.get_free_ram_time(),
                "mean_latency": mean_lat,
            })

    # ----------------------- active-count tracking ---------------------------
    def _record_active_delta(self, delta: int):
        now = self.env.now
        dt = now - self._last_active_t
        if dt > 0:
            self._minute_level_time[self._active_count] += dt
        self._active_count += delta
        self._last_active_t = now

    # ----------------------- redundant (idle-pod) tracking -------------------
    def update_idle_stats(self):
        """Also accumulate idle-container-seconds (idle pod count integrated over
        time) before delegating to the base idle-stats flush. Reuses the base
        flush-before-mutate points, but the running total is never reset at
        warmup, so it stays a pure accumulation from t=0."""
        now = self.env.now
        dt = now - self.last_idle_update
        if dt > 0:
            self.total_redundant_pod_time += dt * len(self.idle_containers)
        super().update_idle_stats()

    def get_redundant_pod_time(self):
        """Total idle-container-seconds (idle warm-pool pods * time idle)."""
        self.update_idle_stats()
        return self.total_redundant_pod_time

    @staticmethod
    def _percentiles_from_hist(hist, total_time):
        if not hist or total_time <= 0:
            return 0, 0
        levels = sorted(hist.keys())
        cum = 0.0
        p95 = p99 = levels[-1]
        thr95 = 0.95 * total_time
        thr99 = 0.99 * total_time
        found_95 = found_99 = False
        for lvl in levels:
            cum += hist[lvl]
            if not found_95 and cum >= thr95:
                p95 = lvl
                found_95 = True
            if not found_99 and cum >= thr99:
                p99 = lvl
                found_99 = True
            if found_95 and found_99:
                break
        return p95, p99

    def _inject_arrivals_exponential(self, count):
        if count <= 0:
            return
        rate_per_second = count / SECONDS_PER_MINUTE
        elapsed = 0.0
        while True:
            gap = random.expovariate(rate_per_second)
            if elapsed + gap >= SECONDS_PER_MINUTE:
                return
            yield self.env.timeout(gap)
            elapsed += gap
            request = Request(next(self.req_id_counter), self.env.now, self.config["request"])
            if self.service_distribution == "traces":
                request.service_time = self.get_next_trace_service_time()
            self.minute_arrivals_count += 1
            if self.env.now >= self.warmup_time:
                self.request_stats["generated"] += 1
            if self.verbose:
                print(f"{self.env.now:.2f} - Request generated: {request}")
            self.env.process(self.handle_request(request))

    # ----------------------- per-minute block / cold tracking ----------------
    def handle_request(self, request):
        """Detect block / cold-start decision at entry time, then delegate.

        In synthetic mode arrivals come from the base request_generator, so the
        per-minute arrival count is tallied here rather than in
        ``_inject_arrivals_exponential``."""
        if not self.trace_driven:
            self.minute_arrivals_count += 1
        chosen = next((c for c in self.idle_containers if c.state == "Idle"), None)
        if chosen is None:
            if self.find_server_for_spawn(request.resource_info) is None:
                self.minute_blocked_count += 1
            else:
                self.minute_cold_start_count += 1
        yield from super().handle_request(request)


# ----------------------------- runner ----------------------------------------
def run_one_repetition(rep_idx: int, seed: int, cfg: dict, trace, n_minutes: int,
                       model, obs_normalizer):
    """Execute a single simulation run and return the system.

    ``trace`` is the per-minute arrival series for trace-driven runs, or None
    for synthetic-arrival runs (in which case ``n_minutes`` sets the tick count).
    """
    global _ACTIVE_TRACKER
    random.seed(seed)
    np.random.seed(seed)
    # Reset Container ID counter so IDs are comparable across runs.
    Container.id_counter = itertools.count()

    env = simpy.Environment()
    system = RLSystem(
        env, cfg,
        distribution=cfg["distribution"],
        trace_per_minute=trace,
        use_ml=USE_ML,
        model=model,
        obs_normalizer=obs_normalizer,
        repetition=rep_idx,
        seed=seed,
        n_minutes=n_minutes,
        verbose=cfg["system"]["verbose"],
    )
    _ACTIVE_TRACKER = system
    for i in range(cfg["system"]["num_servers"]):
        system.add_server(Server(env, f"Server-{i}", cfg["server"]))

    env.process(system.minute_ticker())
    env.process(system.request_generator())
    env.process(system.warmup_reset_process())

    # +1s tail so the final minute_ticker iteration appends its row.
    env.run(until=cfg["system"]["sim_time"] + 1.0)

    # Run-level latency percentiles, stamped onto every minute row of this run
    # (constant per repetition) so the per-minute combined CSV carries them.
    p95_lat, p99_lat = system.get_latency_percentiles()
    for row in system.minute_log:
        row["p95_latency"] = p95_lat
        row["p99_latency"] = p99_lat
    return system


def format_run_summary(system: RLSystem) -> str:
    stats = system.request_stats
    lat = system.latency_stats
    avg_lat = (lat["total_latency"] / lat["count"]) if lat["count"] > 0 else 0.0
    p95_lat, p99_lat = system.get_latency_percentiles()
    bp = (stats["blocked_no_server_capacity"] / stats["generated"]) if stats["generated"] > 0 else 0.0
    idle_vals = [r["idle_timeout"] for r in system.minute_log]
    idle_str = (f"idle[mean/min/max]={np.mean(idle_vals):.1f}/"
                f"{np.min(idle_vals):.1f}/{np.max(idle_vals):.1f}s"
                if idle_vals else "idle=n/a")
    return (f"  rep={system.repetition:>2}  seed={system.seed}  "
            f"gen={stats['generated']}  proc={stats['processed']}  "
            f"block={bp*100:.2f}%  mean_lat={avg_lat:.3f}s  "
            f"p95_lat={p95_lat:.3f}s  p99_lat={p99_lat:.3f}s  "
            f"mean_power={system.get_mean_power_usage():.1f}W  {idle_str}")


def print_run_summary(system: RLSystem):
    print(format_run_summary(system))


# ----------------------------- parallel execution ----------------------------
# Per-worker-process globals, populated once by the pool initializer so the
# (unpicklable) policy is loaded at most once per process rather than per task.
_WORKER_MODEL = None
_WORKER_OBS_NORM = None


def _worker_init(model_path, vecnorm_path, algorithm):
    """Run once per worker process (spawn start method): install the Container
    hooks in this process and, for RL mode, load the policy + VecNormalize
    stats into process-local globals."""
    _install_container_hooks()
    if model_path is None:
        return
    import pickle
    try:
        import torch
        torch.set_num_threads(1)  # avoid CPU oversubscription across workers
    except ImportError:
        pass
    from stable_baselines3 import PPO, SAC
    algo_classes = {"PPO": PPO, "SAC": SAC}
    global _WORKER_MODEL, _WORKER_OBS_NORM
    _WORKER_MODEL = algo_classes[algorithm].load(str(model_path))
    with open(vecnorm_path, "rb") as f:
        _WORKER_OBS_NORM = pickle.load(f)


def _rl_rep_worker(task):
    """Execute one RL-mode repetition; return (rep_idx, minute_log, summary)."""
    rep_idx, seed, cfg, trace, n_minutes = task
    system = run_one_repetition(rep_idx, seed, cfg, trace, n_minutes,
                                _WORKER_MODEL, _WORKER_OBS_NORM)
    return rep_idx, system.minute_log, format_run_summary(system)


def _static_rep_worker(task):
    """Execute one static-mode repetition (looping over timeouts); return
    (rep_idx, minute_rows, summary_rows, summaries)."""
    rep_idx, seed, trace, n_minutes, sim_time, timeouts, combined_static = task
    minute_rows = []
    summary_rows = []
    summaries = []
    for idle_timeout in timeouts:
        cfg = build_config(sim_time, idle_timeout=idle_timeout)
        system = run_one_repetition(rep_idx, seed, cfg, trace, n_minutes, None, None)
        summaries.append((idle_timeout, format_run_summary(system)))
        if combined_static:
            minute_rows.extend(system.minute_log)
        else:
            summary_rows.append(summarize_run(system, idle_timeout))
    return rep_idx, minute_rows, summary_rows, summaries


def _run_repetitions_parallel(worker, tasks, init_args):
    """Map ``worker`` over ``tasks`` across worker processes, blocking until ALL
    finish (barrier) and returning results in task order. Falls back to an
    in-process serial map when only one worker is warranted."""
    n_workers = NUM_WORKERS or min(len(tasks), os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(tasks)))
    if n_workers == 1:
        _worker_init(*init_args)
        return [worker(t) for t in tasks]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init,
                  initargs=init_args) as pool:
        return pool.map(worker, tasks)


def write_log(all_rows, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return csv_path


def write_combined_log(all_rows, n_reps: int) -> Path:
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return write_log(all_rows, LOG_DIR / f"{ts}_multi_x{n_reps}_RL.csv")


def write_combined_static_log(all_rows, n_reps: int) -> Path:
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return write_log(all_rows, LOG_DIR / f"{ts}_multi_x{n_reps}_static.csv")


def summarize_run(system: RLSystem, idle_timeout) -> dict:
    """Collapse one static-mode run into a single whole-run average row.

    Stationary Poisson traffic means per-minute detail is uninformative, so
    every minute_log column is averaged (or taken as a run total) into one row.
    """
    stats = system.request_stats
    rows = system.minute_log
    n = len(rows)
    generated = stats["generated"]
    blocked = stats["blocked_no_server_capacity"]
    cold = stats["container_spawns_initiated"]
    reuse = stats["container_reuses"]
    lat = system.latency_stats
    mean_lat = (lat["total_latency"] / lat["count"]) if lat["count"] > 0 else 0.0
    p95_lat, p99_lat = system.get_latency_percentiles()

    def col_mean(key):
        return (sum(r[key] for r in rows) / n) if n else 0.0

    return {
        "timeout": idle_timeout,
        "repetition": system.repetition,
        "seed": system.seed,
        "sim_time": system.config["system"]["sim_time"],
        "generated": generated,
        "accepted": generated - blocked,
        "blocked": blocked,
        "blocking_prob": (blocked / generated) if generated else 0.0,
        "cold_start": cold,
        "cold_start_ratio": (cold / generated) if generated else 0.0,
        "reuse_ratio": (reuse / generated) if generated else 0.0,
        "mean_warm_pool": col_mean("warm_pool_size"),
        # redundant is now a cumulative idle-pod-seconds column, so averaging it
        # is meaningless; report the time-weighted mean idle pod count instead.
        "mean_redundant": system.get_mean_idle_count(),
        "p95_active": col_mean("p95_active"),
        "p99_active": col_mean("p99_active"),
        "mean_cold_starting": system.get_mean_cold_starting_count(),
        "mean_processing": system.get_mean_processing_count(),
        "mean_idle": system.get_mean_idle_count(),
        "mean_latency": mean_lat,
        "p95_latency": p95_lat,
        "p99_latency": p99_lat,
        "mean_cpu": system.get_mean_cpu_usage(),
        "mean_ram": system.get_mean_ram_usage(),
        "mean_power": system.get_mean_power_usage(),
        "free_ram_time": system.get_free_ram_time(),
    }


def write_summary_log(summary_rows, rep_idx: int, out_dir: Path) -> Path:
    """Write one repetition's idle-timeout summary rows to its own CSV."""
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    csv_path = out_dir / f"{ts}_static_sweep_x{rep_idx + 1}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    return csv_path


# ----------------------------- modes -----------------------------------------
def run_ml_inference():
    """USE_ML=True: RL-controlled idle timeout.

    Arrival source is decided by USE_TRACE:
      * True  -> per-minute counts loaded from TRACE_CSV.
      * False -> synthetic arrivals from variables.py; sim_time taken from
                 variables.py (rounded up to whole minutes).
    """
    if USE_TRACE:
        n_minutes = DAYS_TO_RUN * MINUTES_PER_DAY
        trace = load_trace(TRACE_CSV, n_minutes)
        sim_time = n_minutes * SECONDS_PER_MINUTE
    else:
        raw_sim_time = float(base_config["system"]["sim_time"])
        n_minutes = max(1, (int(raw_sim_time) + SECONDS_PER_MINUTE - 1) // SECONDS_PER_MINUTE)
        sim_time = n_minutes * SECONDS_PER_MINUTE
        trace = None

    cfg = build_config(sim_time)

    # The policy + VecNormalize stats are loaded inside each worker process
    # (see _worker_init); here we only validate the algorithm and paths early.
    algo = ALGORITHM.upper()
    if algo not in {"PPO", "SAC"}:
        raise ValueError(
            f"Unknown ALGORITHM {ALGORITHM!r}; choose from ['PPO', 'SAC']."
        )
    if not Path(f"{MODEL_PATH}.zip").exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}.zip\n"
            f"Re-run dynamic_pool/RL/train.py to regenerate it."
        )
    if not VECNORM_PATH.exists():
        raise FileNotFoundError(
            f"VecNormalize stats not found: {VECNORM_PATH}\n"
            f"Re-run dynamic_pool/RL/train.py to regenerate them."
        )

    print("\n=== dynamic_pool/RL/infer.py — multi-run RL-controlled dynamic pool ===")
    print(f"  USE_ML:           {USE_ML}")
    print(f"  USE_TRACE:        {USE_TRACE}")
    if USE_TRACE:
        print(f"  DAYS_TO_RUN:      {DAYS_TO_RUN} ({n_minutes} minutes)")
    else:
        print(f"  Arrival (config): mean={base_config['request']['arrival_rate_mean']}, "
              f"std={base_config['request']['arrival_rate_std']} "
              f"({base_config['distribution']['arrival-distribution']})")
    print(f"  NUM_REPETITIONS:  {NUM_REPETITIONS}  (BASE_SEED={BASE_SEED})")
    print(f"  sim_time:         {sim_time}s ({n_minutes} minutes)")
    if USE_TRACE:
        print(f"  Trace mean/min:   {trace.mean():.2f} (min={trace.min():.0f}, max={trace.max():.0f})")
    print(f"  Step duration:    {STEP_DURATION}s")
    print(f"  Algorithm:        {ALGORITHM.upper()}")
    print(f"  Checkpoint:       {'best' if USE_BEST else 'final'}")
    print(f"  Idle range:       [{IDLE_MIN}, {IDLE_MAX}]s")
    print(f"  Workers:          {NUM_WORKERS or 'auto'}")
    print(f"  Model:            {MODEL_PATH}.zip")
    print(f"  VecNormalize:     {VECNORM_PATH.name}")
    print(f"  Service dist:     {cfg['distribution']['service-distribution']}")
    print(f"  Spawn dist:       {cfg['distribution']['spawn-distribution']}")

    print("\n--- Repetitions ---")
    tasks = [(rep, BASE_SEED + rep, cfg, trace, n_minutes)
             for rep in range(NUM_REPETITIONS)]
    # Barrier: every repetition completes before we touch a file. pool.map
    # returns results in task (repetition) order, so rows stay correctly slotted.
    results = _run_repetitions_parallel(
        _rl_rep_worker, tasks, (MODEL_PATH, VECNORM_PATH, ALGORITHM.upper())
    )

    all_rows = []
    for _rep_idx, minute_log, summary in results:
        all_rows.extend(minute_log)
        print(summary)

    csv_path = write_combined_log(all_rows, NUM_REPETITIONS)
    print(f"\nCombined per-minute log ({len(all_rows)} rows) written to:")
    print(f"  {csv_path}")


def run_static_sweep():
    """USE_ML=False: static idle-timeout sweep (RL disabled). Each run is
    collapsed to a single whole-run average; all rows are collocated in one
    CSV in logs/dynamic/.

    Arrival source is decided by USE_TRACE:
      * True  -> per-minute counts loaded from TRACE_CSV.
      * False -> synthetic arrivals from variables.py.
    """
    if USE_TRACE:
        n_minutes = DAYS_TO_RUN * MINUTES_PER_DAY
        trace = load_trace(TRACE_CSV, n_minutes)
        sim_time = n_minutes * SECONDS_PER_MINUTE
    else:
        raw_sim_time = float(base_config["system"]["sim_time"])
        # Round up to whole minutes so the minute_ticker covers the full sim_time.
        n_minutes = max(1, (int(raw_sim_time) + SECONDS_PER_MINUTE - 1) // SECONDS_PER_MINUTE)
        sim_time = n_minutes * SECONDS_PER_MINUTE
        trace = None

    arrival_rate = base_config["request"]["arrival_rate_mean"]
    # Special combo: trace-driven + no sweep -> emit a per-minute combined log
    # under the trace's LOG_DIR (mirroring USE_ML=True), suffixed _static.csv.
    combined_static = USE_TRACE and not SWEEP
    out_dir = LOG_DIR if combined_static else STATIC_LOG_DIR / f"arrival_{arrival_rate}"

    if SWEEP:
        timeouts = list(IDLE_TIMEOUTS)
    else:
        timeouts = [base_config["container"]["idle_cpu_timeout"]]

    print("\n=== dynamic_pool/RL/infer.py — static idle-timeout sweep (RL disabled) ===")
    print(f"  USE_ML:           {USE_ML}")
    print(f"  USE_TRACE:        {USE_TRACE}")
    print(f"  SWEEP:            {SWEEP}")
    if USE_TRACE:
        print(f"  DAYS_TO_RUN:      {DAYS_TO_RUN} ({n_minutes} minutes)")
        print(f"  Trace mean/min:   {trace.mean():.2f} (min={trace.min():.0f}, max={trace.max():.0f})")
    else:
        print(f"  Arrival (config): mean={base_config['request']['arrival_rate_mean']}, "
              f"std={base_config['request']['arrival_rate_std']} "
              f"({base_config['distribution']['arrival-distribution']})")
    print(f"  sim_time:         {sim_time}s ({n_minutes} minutes)")
    print(f"  NUM_REPETITIONS:  {NUM_REPETITIONS}  (BASE_SEED={BASE_SEED})")
    print(f"  idle_cpu_timeout: {timeouts}")
    print(f"  Service dist:     {base_config['distribution']['service-distribution']}")
    print(f"  Spawn dist:       {base_config['distribution']['spawn-distribution']}")
    print(f"  Workers:          {NUM_WORKERS or 'auto'}")
    print(f"  Output dir:       {out_dir}")

    tasks = [(rep, BASE_SEED + rep, trace, n_minutes, sim_time, timeouts, combined_static)
             for rep in range(NUM_REPETITIONS)]
    # Barrier: all repetitions finish before any file is written; results come
    # back in repetition order so per-rep CSVs and combined rows stay aligned.
    results = _run_repetitions_parallel(_static_rep_worker, tasks, (None, None, None))

    written_paths = []
    all_minute_rows = []
    for rep_idx, minute_rows, summary_rows, summaries in results:
        print(f"\n=== Repetition {rep_idx + 1}/{NUM_REPETITIONS}  (seed={BASE_SEED + rep_idx}) ===")
        for idle_timeout, summary in summaries:
            print(f"--- idle_cpu_timeout = {idle_timeout}s ---")
            print(summary)
        if combined_static:
            all_minute_rows.extend(minute_rows)
        else:
            csv_path = write_summary_log(summary_rows, rep_idx, out_dir)
            written_paths.append(csv_path)
            print(f"  -> rep {rep_idx + 1} summary written to {csv_path}")

    if combined_static:
        csv_path = write_combined_static_log(all_minute_rows, NUM_REPETITIONS)
        print(f"\nCombined per-minute log ({len(all_minute_rows)} rows) written to:")
        print(f"  {csv_path}")
    else:
        print(f"\n{len(written_paths)} per-repetition summary CSV(s) written to:")
        for p in written_paths:
            print(f"  {p}")


def main():
    _install_container_hooks()
    if USE_ML:
        run_ml_inference()
    else:
        run_static_sweep()


if __name__ == "__main__":
    main()
