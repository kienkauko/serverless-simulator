"""
Multi-run ML-driven serverless simulator.

Mirrors main_ML_proactive.py but executes ``NUM_REPETITIONS`` independent
simulation runs, each seeded with a different random seed, so the per-minute
metrics can be aggregated across runs for statistical analysis (mean / std /
CI etc.). All runs share the same trace, model, and config.

Per-minute log fields (one row per (repetition, minute)):
  - repetition       : run index (0..NUM_REPETITIONS-1)
  - seed             : random seed used for this run
  - minute           : minute index within the run
  - time             : env.now at the end of the minute (s)
  - arrival          : arrivals injected during the minute
  - accepted         : arrivals not rejected for lack of capacity
  - warm_pool_size   : len(idle_containers) at end of minute
  - cold_hit         : cumulative reactive cold starts since run start
                       (running total; excludes pool-resize spawns)
  - redundant        : cumulative idle-container-seconds since run start —
                       integral of (free/idle warm-pool pod count) over time
                       (a wasted-capacity indicator, in pod*s)
  - energy           : cumulative energy since warmup end (W*s)
  - ram_area         : cumulative RAM time-area since warmup end (%*s)
  - cpu_area         : cumulative CPU time-area since warmup end (%*s)
  - free_ram_time    : cumulative warm/idle RAM-seconds since warmup end —
                       RAM held by warm-pool pods while Idle (cold-start /
                       processing RAM excluded); a wasted-RAM indicator
  - mean_latency     : running mean end-to-end latency of finished requests (s)
  - p95_latency      : whole-run 95th-pctile end-to-end latency (s); constant
                       per repetition (same value on every minute row of a run)
  - p99_latency      : whole-run 99th-pctile end-to-end latency (s); constant
                       per repetition (same value on every minute row of a run)
"""

import csv
import json
import multiprocessing as mp
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import math
import joblib
import numpy as np
import simpy

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from variables import config as base_config
from Server import Server
from fixed_pool.System import System
from Request import Request
from fixed_pool.Container import Container

# ----------------------------- knobs -----------------------------------------
USE_ML = True
DAYS_TO_RUN = 30
NUM_REPETITIONS = 10
BASE_SEED = 42
# Worker processes used to run repetitions in parallel (1 -> serial). Honors the
# SIM_NUM_WORKERS env override so run_all_cases.py can drive batches; defaults to
# serial to preserve standalone behavior.
NUM_WORKERS = int(os.environ.get("SIM_NUM_WORKERS", "1"))

# Set once (first MLSystem instance constructed); used by monkey-patched
# Container hooks to attribute active-count transitions to the live system.
_ACTIVE_TRACKER = None
_CONTAINER_HOOKS_INSTALLED = False


def _install_container_hooks():
    """Wrap Container.assign_request / release_request to record Idle<->Active
    transitions in the current MLSystem's per-minute active-count histogram.
    Patching once at module level is safe across repetitions."""
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # arrival_data/ and logs/ live at the repo root, not inside fixed_pool/
ML_TESTS_DIR = REPO_ROOT / "arrival_data" / "ML_tests"

# Select the arrival trace; MODEL_DIR and LOG_DIR are derived from it.
# Honors the SIM_TRACE_NAME env override so run_all_cases.py can drive batches.
TRACE_NAME = os.environ.get("SIM_TRACE_NAME", "non_station.csv")

_TRACE_CONFIG = {
    "non_station.csv": {
        "model": "best_model_non_station",
        "log_subdir": "non_station",
    },
    "day_night.csv": {
        "model": "best_model_day_night",
        "log_subdir": "day_night",
    },
}

if TRACE_NAME not in _TRACE_CONFIG:
    raise ValueError(
        f"Unknown TRACE_NAME {TRACE_NAME!r}; "
        f"expected one of {sorted(_TRACE_CONFIG)}."
    )

TRACE_CSV = ML_TESTS_DIR / TRACE_NAME
MODEL_DIR = ML_TESTS_DIR / _TRACE_CONFIG[TRACE_NAME]["model"]
LOG_DIR = REPO_ROOT / "logs" / _TRACE_CONFIG[TRACE_NAME]["log_subdir"]

SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440


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
            f"Trace has only {len(series)} minutes, need {n_minutes}."
        )
    return series[:n_minutes]


def load_model_metadata(model_dir: Path) -> dict:
    return json.loads((model_dir / "metadata.json").read_text())


def _build_quantile_lstm():
    """Architecture must match train_predictors.get_lstm_model()."""
    import torch.nn as nn

    class QuantileLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            x, _ = self.lstm2(x)
            x = self.dropout2(x)
            x = x[:, -1, :]
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return QuantileLSTM()


def make_predictor(model_dir: Path):
    bundle = joblib.load(model_dir / "bundle.joblib")
    scaler = bundle["scaler"]
    lookback = bundle["lookback"]
    bias_offset = float(bundle.get("bias_offset", 0.0))

    if bundle["model_type"] == "LSTM":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # `keras_path` is a legacy key — it actually points to the PyTorch state_dict (lstm.pth).
        weights_path = model_dir / bundle.get("keras_path", "lstm.pth")
        model = _build_quantile_lstm().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        def predict(window):
            arr = np.asarray(window, dtype=float).reshape(-1, 1)
            scaled = scaler.transform(arr).ravel().reshape(1, lookback, 1)
            with torch.no_grad():
                scaled_t = torch.tensor(scaled, dtype=torch.float32).to(device)
                scaled_pred = model(scaled_t).item() + bias_offset
            return float(scaler.inverse_transform([[scaled_pred]])[0, 0])
    else:
        model = bundle["model"]
        booster = getattr(model, "booster_", None)

        def predict(window):
            arr = np.asarray(window, dtype=float).reshape(-1, 1)
            scaled = scaler.transform(arr).ravel().reshape(1, lookback)
            raw = booster.predict(scaled) if booster is not None else model.predict(scaled)
            scaled_pred = float(np.ravel(raw)[0]) + bias_offset
            return float(scaler.inverse_transform([[scaled_pred]])[0, 0])

    return predict, lookback, bundle["model_type"]


# ----------------------------- ML-aware System -------------------------------
class MLSystem(System):
    """System extension: trace-driven Poisson arrivals + ML-driven warm pool."""

    def __init__(self, env, config, distribution, trace_per_minute,
                 use_ml, predictor, lookback, repetition, seed, verbose=False):
        super().__init__(env, config, distribution, verbose=verbose)
        self.trace = np.asarray(trace_per_minute, dtype=float)
        self.n_minutes = len(self.trace)
        self.use_ml = use_ml
        self.predictor = predictor
        self.lookback = lookback
        self.repetition = repetition
        self.seed = seed

        resource = max(config["request"]["ram_demand"], config["request"]["cpu_demand"])
        self.max_pool_capacity = int(np.floor(100 / resource) * config["system"]["num_servers"])

        if self.service_distribution == "traces":
            self.mean_service_time = float(np.mean(self.service_trace_times))
        else:
            self.mean_service_time = 1.0 / float(config["request"]["service_rate"])

        self.pool_history = []

        # Per-minute counters
        self.minute_arrivals_count = 0
        self.minute_cold_start_count = 0
        self.minute_blocked_count = 0
        self.minute_log = []

        # Cumulative (accumulated) metrics — running totals across the run.
        # cumulative_cold_hit: reactive cold starts summed over all minutes.
        # total_redundant_pod_time: idle-container-seconds, i.e. the integral of
        # (free/idle warm-pool pod count) over time. Flushed via the same
        # flush-before-mutate points as the idle-RAM area (see
        # update_idle_ram_stats override below).
        self.cumulative_cold_hit = 0
        self.total_redundant_pod_time = 0.0
        self._last_redundant_update = 0.0

        # Active-container time-weighted histogram (per-minute, reset each tick).
        # _active_count: containers currently in "Active" state (processing).
        # _minute_level_time[level] += seconds spent at that level during the minute.
        self._active_count = 0
        self._last_active_t = 0.0
        self._minute_level_time = defaultdict(float)

    # ----------------------- arrival generation ------------------------------
    def request_generator(self):
        if False:
            yield
        return

    def minute_ticker(self):
        # Target computed last iter for the minute that's about to start.
        # Used to apply the deferred shrink half of the asymmetric resize.
        pending_target = None
        for m in range(self.n_minutes):
            minute_start = self.env.now
            expected_rate = max(0.0, float(self.trace[m]))

            if self.use_ml:
                # Deferred shrink: trim down to the target predicted at iter m-1
                # (which targeted this minute m). Skip on the first iter.
                if pending_target is not None and pending_target < len(self.idle_containers):
                    self._shrink_warm_pool(pending_target)

                # Predict for the *next* minute (m+1) and grow proactively now so
                # spawn time overlaps minute m instead of biting into m+1.
                if m >= self.lookback:
                    window = self.trace[m - self.lookback:m]
                    predicted = max(0.0, float(self.predictor(window)))
                    # Offered load (Erlangs) a = predicted_traffic * mean_service_time.
                    # Provision pods via the square-root staffing rule: a + 2*sqrt(a),
                    # which sizes the safety margin to demand variability rather than
                    # a flat percentage headroom.
                    a = predicted * self.mean_service_time / SECONDS_PER_MINUTE
                    next_target = math.ceil(a + 2.0 * math.sqrt(a))
                    next_target = max(0, min(next_target, self.max_pool_capacity))
                    if next_target > len(self.idle_containers):
                        self._grow_warm_pool(next_target)
                    pending_target = next_target
                    self.pool_history.append({
                        "minute": m + 1,
                        "predicted_count": predicted,
                        "target_pool": next_target,
                        "pool_after": len(self.idle_containers),
                    })

            # Reset per-minute counters and snapshot currently-idle pods.
            self.minute_arrivals_count = 0
            self.minute_cold_start_count = 0
            self.minute_blocked_count = 0

            # Reset active-count histogram for this minute. _active_count carries
            # over from previous minute (it reflects the live state).
            self._minute_level_time = defaultdict(float)
            self._last_active_t = minute_start

            if expected_rate > 0:
                self.env.process(self._inject_arrivals_exponential(expected_rate))

            yield self.env.timeout(SECONDS_PER_MINUTE)

            # Accumulate cold starts into the running total for this minute.
            self.cumulative_cold_hit += self.minute_cold_start_count

            # Flush time-weighted accumulators up to "now" before reading them.
            self.update_resource_stats()
            mean_lat = (self.latency_stats['total_latency'] / self.latency_stats['count']
                        if self.latency_stats['count'] > 0 else 0.0)

            # Flush trailing segment of the active-count histogram for this minute.
            now = self.env.now
            trailing = now - self._last_active_t
            if trailing > 0:
                self._minute_level_time[self._active_count] += trailing
            self._last_active_t = now
            p95_active, p99_active = self._percentiles_from_hist(
                self._minute_level_time, SECONDS_PER_MINUTE
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
                "energy": self.total_energy_usage,
                "ram_area": self.total_ram_usage_area,
                "cpu_area": self.total_cpu_usage_area,
                "free_ram_time": self.get_free_ram_time(),
                "mean_latency": mean_lat,
            })

            if (m + 1) % MINUTES_PER_DAY == 0:
                day = (m + 1) // MINUTES_PER_DAY
                total_days = self.n_minutes // MINUTES_PER_DAY
                gen = self.request_stats["generated"]
                proc = self.request_stats["processed"]
                blk = self.request_stats["blocked_no_server_capacity"]
                print(f"  [rep={self.repetition} seed={self.seed}] "
                      f"day {day}/{total_days} done  ",
                      flush=True)

    # ----------------------- active-count tracking ---------------------------
    def _record_active_delta(self, delta: int):
        """Called from monkey-patched Container hooks on Idle<->Active flips."""
        now = self.env.now
        dt = now - self._last_active_t
        if dt > 0:
            self._minute_level_time[self._active_count] += dt
        self._active_count += delta
        self._last_active_t = now

    # ----------------------- redundant (idle-pod) tracking -------------------
    def update_idle_ram_stats(self):
        """Accumulate idle-container-seconds (free/idle warm-pool pod count
        integrated over time) before delegating to the base RAM-area flush.

        Reuses the base class's flush-before-mutate contract: every call site
        that flushes idle RAM also flushes the redundant pod-time here, so any
        change to warm-pool membership or a pooled pod's Idle state attributes
        the elapsed interval to the idle pod count held during it."""
        now = self.env.now
        idle_count = sum(1 for c in self.idle_containers if c.state == "Idle")
        self.total_redundant_pod_time += (now - self._last_redundant_update) * idle_count
        self._last_redundant_update = now
        super().update_idle_ram_stats()

    def get_redundant_pod_time(self):
        """Total idle-container-seconds (free warm-pool pods * time idle)."""
        self.update_idle_ram_stats()
        return self.total_redundant_pod_time

    @staticmethod
    def _percentiles_from_hist(hist, total_time):
        """Return (p95, p99): smallest active-level whose cumulative time-share
        within the minute reaches the given quantile. Time-weighted, exact."""
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

    # ----------------------- per-minute block tracking ------------------------
    def handle_request(self, request):
        """Detect a block at decision time (before super yields), then delegate."""
        chosen = next((c for c in self.idle_containers if c.state == "Idle"), None)
        if chosen is None and self.find_server_for_spawn(request.resource_info) is None:
            self.minute_blocked_count += 1
        yield from super().handle_request(request)

    # ----------------------- cold-start instrumentation ----------------------
    def spawn_container_process(self, server, request, is_pre_warm=False):
        if not is_pre_warm:
            self.minute_cold_start_count += 1
        container = yield from super().spawn_container_process(
            server, request, is_pre_warm=is_pre_warm
        )
        return container

    # ----------------------- warm-pool resizing ------------------------------
    def _shrink_warm_pool(self, target: int):
        current = len(self.idle_containers)
        if target >= current:
            return
        n_remove = current - target
        self.update_idle_ram_stats()   # flush before idle pods leave the pool
        idle_pods = [c for c in self.idle_containers if c.state == "Idle"]
        busy_pods = [c for c in self.idle_containers if c.state != "Idle"]
        removed = 0
        for c in idle_pods:
            if removed >= n_remove:
                break
            self.idle_containers.remove(c)
            c.release_resources()
            removed += 1
        for c in busy_pods:
            if removed >= n_remove:
                break
            self.idle_containers.remove(c)
            removed += 1
        if self.verbose:
            print(f"{self.env.now:.2f} - Pool shrink {current} -> {target} "
                  f"(removed {removed})")

    def _grow_warm_pool(self, target: int):
        current = len(self.idle_containers)
        if target <= current:
            return
        need = target - current
        for _ in range(need):
            template = Request(9999, self.env.now, self.config["request"])
            server = self.find_server_for_spawn(template.resource_info)
            if server is None:
                if self.verbose:
                    print(f"{self.env.now:.2f} - Pool grow aborted, no capacity.")
                break
            self.allocate_server(server, template, is_pre_warm=True)
            self.env.process(self._spawn_warm_async(server, template))
        if self.verbose:
            print(f"{self.env.now:.2f} - Pool grow {current} -> {target}")

    def _spawn_warm_async(self, server, request):
        spawned = yield self.env.process(
            self.spawn_container_process(server, request, is_pre_warm=True)
        )
        if spawned is not None:
            self.update_idle_ram_stats()   # flush before this warm pod joins the pool
            self.idle_containers.append(spawned)


# ----------------------------- runner ----------------------------------------
def build_config(lookback_minutes: int) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base_config.items()}
    cfg["system"] = dict(base_config["system"])
    cfg["system"]["warmup_time"] = lookback_minutes * SECONDS_PER_MINUTE
    cfg["system"]["sim_time"] = DAYS_TO_RUN * MINUTES_PER_DAY * SECONDS_PER_MINUTE
    return cfg


def run_one_repetition(rep_idx: int, seed: int, cfg: dict, trace: np.ndarray,
                       predictor, lookback: int):
    """Execute a single simulation run and return its minute_log rows."""
    global _ACTIVE_TRACKER
    random.seed(seed)
    np.random.seed(seed)
    # Reset Container ID counter so IDs are comparable across runs.
    import itertools
    Container.id_counter = itertools.count()

    env = simpy.Environment()
    system = MLSystem(
        env, cfg,
        distribution=cfg["distribution"],
        trace_per_minute=trace,
        use_ml=USE_ML,
        predictor=predictor,
        lookback=lookback,
        repetition=rep_idx,
        seed=seed,
        verbose=cfg["system"]["verbose"],
    )
    _ACTIVE_TRACKER = system
    for i in range(cfg["system"]["num_servers"]):
        system.add_server(Server(env, f"Server-{i}", cfg["server"]))

    pre_warm_done = env.process(system.pre_warm())

    def start_all():
        yield pre_warm_done
        env.process(system.minute_ticker())

    env.process(start_all())
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


def format_run_summary(system: MLSystem) -> str:
    stats = system.request_stats
    lat = system.latency_stats
    avg_lat = (lat["total_latency"] / lat["count"]) if lat["count"] > 0 else 0.0
    p95_lat, p99_lat = system.get_latency_percentiles()
    bp = (stats["blocked_no_server_capacity"] / stats["generated"]) if stats["generated"] > 0 else 0.0
    return (f"  rep={system.repetition:>2}  seed={system.seed}  "
            f"gen={stats['generated']}  proc={stats['processed']}  "
            f"block={bp*100:.2f}%  mean_lat={avg_lat:.3f}s  "
            f"p95_lat={p95_lat:.3f}s  p99_lat={p99_lat:.3f}s  "
            f"mean_power={system.get_mean_power_usage():.1f}W")


# ----------------------------- parallel execution ----------------------------
# Per-worker-process predictor cache: the torch/joblib closure is not picklable,
# so each spawned worker loads its own copy once (in _worker_init) rather than
# receiving it through the task tuple.
_WORKER_PREDICTOR = None  # (predict_fn, lookback, model_type) or None


def _worker_init():
    """Run once per worker process (spawn start method): install the Container
    hooks in this process and load the arrival predictor into a process-local
    global so every repetition on this worker reuses it."""
    _install_container_hooks()
    try:
        import torch
        torch.set_num_threads(1)  # avoid CPU oversubscription across workers
    except ImportError:
        pass
    global _WORKER_PREDICTOR
    if USE_ML:
        _WORKER_PREDICTOR = make_predictor(MODEL_DIR)


def _rep_worker(task):
    """Execute one repetition; return (rep_idx, minute_log, summary)."""
    rep_idx, seed, cfg, trace, lookback = task
    predictor = _WORKER_PREDICTOR[0] if (USE_ML and _WORKER_PREDICTOR) else None
    system = run_one_repetition(rep_idx, seed, cfg, trace, predictor, lookback)
    return rep_idx, system.minute_log, format_run_summary(system)


def _run_repetitions_parallel(tasks):
    """Map repetitions across worker processes (barrier: all finish before the
    caller writes any file). Falls back to an in-process serial map when only one
    worker is warranted, preserving the original single-process behavior."""
    n_workers = NUM_WORKERS or min(len(tasks), os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(tasks)))
    if n_workers == 1:
        _worker_init()
        return [_rep_worker(t) for t in tasks]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        return pool.map(_rep_worker, tasks)


def write_combined_log(all_rows, n_reps: int) -> Path:
    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"{ts}_multi_x{n_reps}_SR.csv"
    fields = [
        "repetition", "seed", "minute", "time",
        "arrival", "accepted", "warm_pool_size",
        "cold_hit", "redundant",
        "p95_active", "p99_active",
        "energy", "ram_area", "cpu_area", "free_ram_time",
        "mean_latency", "p95_latency", "p99_latency",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    return csv_path


def main():
    _install_container_hooks()
    n_minutes = DAYS_TO_RUN * MINUTES_PER_DAY
    trace = load_trace(TRACE_CSV, n_minutes)

    predictor = None
    lookback = 0
    model_type = "N/A"
    if USE_ML:
        predictor, lookback, model_type = make_predictor(MODEL_DIR)
    else:
        try:
            meta = load_model_metadata(MODEL_DIR)
            lookback = int(meta.get("lookback", 0))
        except FileNotFoundError:
            lookback = 0

    cfg = build_config(lookback)

    print("\n=== multi_ML_proactive.py ===")
    print(f"  USE_ML:           {USE_ML} (model={model_type})")
    print(f"  DAYS_TO_RUN:      {DAYS_TO_RUN} ({n_minutes} minutes)")
    print(f"  NUM_REPETITIONS:  {NUM_REPETITIONS}  (BASE_SEED={BASE_SEED})")
    print(f"  Lookback:         {lookback} min  -> warmup = {lookback*SECONDS_PER_MINUTE}s")
    print(f"  sim_time:         {cfg['system']['sim_time']}s")
    print(f"  Trace mean/min:   {trace.mean():.2f} (min={trace.min():.0f}, max={trace.max():.0f})")
    print(f"  Service dist:     {cfg['distribution']['service-distribution']}")
    print(f"  Spawn dist:       {cfg['distribution']['spawn-distribution']}")
    print(f"  Workers:          {NUM_WORKERS}")

    print("\n--- Repetitions ---")
    tasks = [(rep, BASE_SEED + rep, cfg, trace, lookback)
             for rep in range(NUM_REPETITIONS)]
    # Barrier: every repetition completes before we touch a file. The parallel
    # map returns results in task (repetition) order, so rows stay slotted.
    results = _run_repetitions_parallel(tasks)

    all_rows = []
    for _rep_idx, minute_log, summary in results:
        all_rows.extend(minute_log)
        print(summary)

    csv_path = write_combined_log(all_rows, NUM_REPETITIONS)
    print(f"\nCombined per-minute log ({len(all_rows)} rows) written to:")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
