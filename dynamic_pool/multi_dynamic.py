"""Dynamic-pool idle-timeout control — static / analytical / RL, one harness.

A single trace-replay harness with three control modes, selected by USE_ML /
USE_RL:

  * STATIC   (USE_ML=False): every container keeps the fixed idle timeout from
    variables.py (``container.idle_cpu_timeout``); no per-minute control.
  * ANALYSIS (USE_ML=True, USE_RL=False): the analytical controller from
    ``controller.py`` predicts next-minute arrivals, solves the SLA-constrained
    resource-minimal idle timeout (min E[R] s.t. E[Br] <= SLA) from the 1D
    warm-pool birth-death approximation, and applies it each minute.
  * RL       (USE_ML=True, USE_RL=True): the trained PPO/SAC policy from
    ``dynamic_pool/RL/`` maps the per-minute observation to an idle timeout.

Each minute the new timeout always governs *upcoming* containers (read on
spawn). ANALYSIS additionally retimes the *currently* idle pool when
APPLY_TO_IDLE is set; RL never retimes existing containers, so decisions affect
only containers spawned afterwards — exactly how the policy was trained. Both
go through the shared ``LoggingSystem._apply_idle_timeout``.

Both dynamic policies duck-type ``predict(obs) -> (action, state)``:
``LoggingSystem.minute_ticker`` calls it once per minute with
obs = [arrivals_last_min, warm_pool, cold_hits, ram_util] and maps the returned
action in [-1, 1] back to a timeout in [IDLE_MIN, IDLE_MAX] (bounds and obs
match dynamic_pool/RL/train.py, so the trained checkpoint plugs in directly).
The arrival forecaster (ANALYSIS/PREDICTOR="ml") is the same trained per-trace
bundle ``multi_ML_proactive_SR.make_predictor`` returns — it predicts the
next-minute arrival *count* (incoming traffic), nothing else.

Knobs:
  USE_ML / USE_RL Select the control mode (see above); honor SIM_USE_ML /
                  SIM_USE_RL env overrides.
  ALGORITHM       RL only: "PPO" | "SAC" (must match the trained checkpoint).
  USE_BEST        RL only: best eval checkpoint vs final-step policy.
  PREDICTOR       ANALYSIS only: "ml" | "last" | "ewma" | "oracle" — next-minute forecast.
                  "ml" loads the same trained per-trace LSTM/LightGBM bundle
                  that multi_ML_proactive.py uses (make_predictor, identical
                  trace[m-lookback:m] window convention), so the analytical
                  and pool-resizing strategies consume the same forecast.
                  "oracle" reads the true trace value (perfect-prediction
                  upper bound); "last"/"ewma" use observed counts only.
  APPLY_TO_IDLE   True  -> each decision also retimes the eviction deadline
                  of every currently idle container to idle_since + timeout
                  (and updates the per-instance setpoint of live containers),
                  so the pool tracks the setpoint immediately.
                  False -> the new timeout only affects containers that go
                  idle after the decision (original lagging behavior).
  WARMUP_MINUTES  Leading transient discarded from the aggregate metrics for
                  non-ML predictors. For PREDICTOR="ml" the warmup is forced
                  to the model's lookback instead (the predictor cannot form a
                  full window before then), matching multi_ML_proactive.py so
                  the steady-state numbers are comparable. The per-minute CSV
                  still carries every minute (filter ``minute >= warmup`` for
                  the steady-state slice).
  LATENCY_SLA     E[Br] constraint in seconds; must exceed E[Bw].
  RESOURCE        "ram" | "cpu" — which per-container demand fills r_t/r_a/r_w
                  in the E[R] objective.

Output: one combined per-minute CSV in ``logs/<trace>/`` suffixed by the mode —
``_static.csv`` / ``_analysis.csv`` / ``_rl.csv``.

Run:  python dynamic_pool/multi_dynamic.py
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
_REPO_ROOT = _HERE.parent
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from variables import config as base_config
from Server import Server
from Request import Request
from dynamic_pool.System import System, _load_processing_profile
from dynamic_pool.Container import Container
from controller import solve_optimal_timeout, expected_latency

# ----------------------------- knobs -----------------------------------------
# TRACE_NAME / NUM_WORKERS / USE_ML / USE_RL honor env overrides (SIM_TRACE_NAME /
# SIM_NUM_WORKERS / SIM_USE_ML / SIM_USE_RL) so run_all_cases.py can drive batches;
# defaults preserve standalone behavior.
TRACE_NAME = os.environ.get("SIM_TRACE_NAME", "non_station.csv")   # "non_station.csv" or "day_night.csv"
DAYS_TO_RUN = 30
NUM_REPETITIONS = 10
BASE_SEED = 42
NUM_WORKERS = int(os.environ.get("SIM_NUM_WORKERS", "4"))

# Control mode (master switch):
#   USE_ML = False -> STATIC: every container uses the fixed idle timeout from
#       variables.py (container.idle_cpu_timeout); no per-minute control.
#   USE_ML = True  -> DYNAMIC: a policy sets the idle timeout each minute and it
#       is applied to current + upcoming containers (see APPLY_TO_IDLE). The
#       policy is selected by USE_RL:
#         USE_RL = True  -> trained RL checkpoint (PPO/SAC) from dynamic_pool/RL/.
#         USE_RL = False -> analytical controller (PREDICTOR-driven, below).
USE_ML = os.environ.get("SIM_USE_ML", "True").strip().lower() in ("1", "true", "yes")
USE_RL = os.environ.get("SIM_USE_RL", "False").strip().lower() in ("1", "true", "yes")
ALGORITHM = "PPO"         # RL only: "PPO" | "SAC" (must match the trained checkpoint)
USE_BEST = True           # RL only: best eval checkpoint vs final-step policy

PREDICTOR = "ml"          # analytical only: "ml" | "last" | "ewma" | "oracle"
EWMA_ALPHA = 0.3          # weight of the newest observation (PREDICTOR="ewma")
APPLY_TO_IDLE = True      # retime currently idle containers on each decision
WARMUP_MINUTES = 120      # transient discarded (non-ML); analytical-ML uses its lookback
LATENCY_SLA = 4.0         # E[Br] constraint (s); must exceed E[Bw] (~2.1s)
RESOURCE = "ram"          # "ram" or "cpu" objective in E[R]
TIMEOUT_GRID_STEP = 0.25  # theta search granularity (s of idle timeout)

# ----------------------------- constants -------------------------------------
STEP_DURATION = 60.0      # decision interval (s) — also the per-minute tick
IDLE_MIN = 1.0            # action->timeout bounds; MUST match dynamic_pool/RL/train.py
IDLE_MAX = 60.0
SECONDS_PER_MINUTE = 60
MINUTES_PER_DAY = 1440

TRACE_CSV = _REPO_ROOT / "arrival_data" / "ML_tests" / TRACE_NAME
LOG_DIR = _REPO_ROOT / "logs" / TRACE_NAME.replace(".csv", "")

# Trained RL checkpoint paths (USE_RL=True), matching dynamic_pool/RL/train.py's
# naming: {algo}_idle_timeout_{trace_stem}[_best]/best_model(.zip) + best_vecnorm.pkl.
_RL_DIR = _REPO_ROOT / "dynamic_pool" / "RL"
_RL_RUN_STEM = f"{ALGORITHM.lower()}_idle_timeout_{TRACE_NAME.replace('.csv', '')}"
if USE_BEST:
    RL_MODEL_PATH = _RL_DIR / f"{_RL_RUN_STEM}_best" / "best_model"
    RL_VECNORM_PATH = _RL_DIR / f"{_RL_RUN_STEM}_best" / "best_vecnorm.pkl"
else:
    RL_MODEL_PATH = _RL_DIR / _RL_RUN_STEM
    RL_VECNORM_PATH = _RL_DIR / f"{_RL_RUN_STEM}_vecnorm.pkl"

# Static-mode idle timeout: fixed for every container, taken from variables.py.
STATIC_IDLE_TIMEOUT = base_config["container"].get(
    "idle_timeout", base_config["container"].get("idle_cpu_timeout", 0))

LOG_FIELDS = [
    "repetition", "seed", "minute", "time",
    "arrival", "accepted", "warm_pool_size",
    "cold_hit", "redundant",
    "p95_active", "p99_active",
    "idle_timeout",
    "energy", "ram_area", "cpu_area", "free_ram_time",
    "mean_latency", "p95_latency", "p99_latency",
]


# ----------------------------- container hooks -------------------------------
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


def build_config(sim_time: float, warmup_minutes: int = 0) -> dict:
    cfg = copy.deepcopy(base_config)
    cfg["system"]["warmup_time"] = warmup_minutes * SECONDS_PER_MINUTE
    cfg["system"]["verbose"] = False
    cfg["system"]["sim_time"] = sim_time
    return cfg


# Lazily-loaded (per process) trained arrival predictor, shared with
# multi_ML_proactive.py: (predict_fn, lookback, model_type).
_ML_PREDICTOR = None


def _get_ml_predictor():
    """Load the per-trace arrival-traffic forecaster (LSTM/LightGBM bundle)
    exactly as multi_ML_proactive_SR does, cached per process (the torch/joblib
    closure is not picklable, so each worker loads its own copy on first use).

    The bundle predicts the next-minute arrival *count* from a window of past
    per-minute counts — i.e. it forecasts incoming traffic, which is what the
    analytical controller's offered-load (Erlang) estimate needs."""
    global _ML_PREDICTOR
    if _ML_PREDICTOR is None:
        import fixed_pool.multi_ML_proactive_SR as mlp
        model_dir = mlp.ML_TESTS_DIR / mlp._TRACE_CONFIG[TRACE_NAME]["model"]
        _ML_PREDICTOR = mlp.make_predictor(model_dir)
    return _ML_PREDICTOR


# Lazily-loaded (per process) trained RL policy + its VecNormalize stats, for
# USE_RL=True. SB3 policies and VecNormalize are not picklable across the spawn
# boundary, so each worker loads its own copy on first use.
_RL_POLICY = None  # (model, vecnormalize) or None


def _get_rl_policy():
    """Load the trained PPO/SAC checkpoint and the matching VecNormalize stats
    from dynamic_pool/RL/ (same files dynamic_pool/RL/train.py produces). Cached
    per process. The policy exposes ``predict(obs, deterministic=True) ->
    (action, state)``, identical to the analytical controller's interface."""
    global _RL_POLICY
    if _RL_POLICY is None:
        import pickle
        from stable_baselines3 import PPO, SAC
        algo_cls = {"PPO": PPO, "SAC": SAC}[ALGORITHM.upper()]
        model = algo_cls.load(str(RL_MODEL_PATH))
        with open(RL_VECNORM_PATH, "rb") as f:
            vecnorm = pickle.load(f)
        _RL_POLICY = (model, vecnorm)
    return _RL_POLICY


# ----------------------------- model parameters ------------------------------
def resolve_model_params(cfg: dict) -> dict:
    """Pull E[Bw], E[Bc] and the per-state resource demands from the config,
    honoring the trace-fitted means when the distributions are trace-driven."""
    dist = cfg["distribution"]
    req = cfg["request"]

    profile = None
    if dist["service-distribution"] == "traces" or dist["spawn-distribution"] == "trace":
        profile = _load_processing_profile(
            str(_REPO_ROOT / "traces" / "measured_traces" / "profile.csv"))

    if dist["service-distribution"] == "traces":
        e_bw = profile["service-time"][0]
    else:
        e_bw = 1.0 / req["service_rate"]

    if dist["spawn-distribution"] == "trace":
        e_bc = profile["cold-start"][0]
    else:
        e_bc = cfg["container"]["spawn_time_mean"]

    if RESOURCE == "ram":
        r_t, r_a, r_w = req["cold_start_ram"], req["ram_demand"], req["warm_ram"]
    elif RESOURCE == "cpu":
        r_t, r_a, r_w = req["cold_start_cpu"], req["cpu_demand"], req["warm_cpu"]
    else:
        raise ValueError(f"Unknown RESOURCE {RESOURCE!r}; expected 'ram' or 'cpu'.")

    if LATENCY_SLA <= e_bw:
        raise ValueError(
            f"LATENCY_SLA={LATENCY_SLA}s is not achievable: it must exceed "
            f"E[Bw]={e_bw:.3f}s (every request pays the service time)."
        )
    return {"e_bw": e_bw, "e_bc": e_bc, "r_t": r_t, "r_a": r_a, "r_w": r_w}


# ----------------------------- logging System --------------------------------
class LoggingSystem(System):
    """Dynamic-pool System extension with per-minute metric logging.

    Trace-driven: arrivals are injected per minute from the trace; the
    idle_timeout is controlled by the duck-typed ``model.predict`` policy.
    """

    def __init__(self, env, config, distribution, trace_per_minute,
                 use_ml, model, obs_normalizer, repetition, seed,
                 n_minutes=None, verbose=False, retime_idle=False):
        super().__init__(env, config, distribution, verbose=verbose)
        # When True, a new setpoint also retimes the *currently* idle pool
        # (ANALYSIS + APPLY_TO_IDLE). RL leaves it False so decisions affect only
        # containers spawned afterwards, exactly as the policy was trained.
        self.retime_idle = retime_idle
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

        # Observation tracking (running tallies that the base System keeps).
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
        """One iteration per simulated minute: query the controller, inject
        trace arrivals, then accumulate per-minute metrics."""
        obs = np.zeros(4, dtype=np.float32)
        span = IDLE_MAX - IDLE_MIN

        for m in range(self.n_minutes):
            minute_start = self.env.now

            # ------------- controller decision -------------
            # DYNAMIC modes (RL or analytical): the policy maps the observation
            # to an action in [-1, 1], which we scale to a timeout and apply via
            # _apply_idle_timeout (upcoming pods always; current idle pool only
            # for ANALYSIS+APPLY_TO_IDLE). STATIC mode (use_ml False) skips this:
            # the configured idle timeout stays fixed for the whole run.
            if self.use_ml and self.model is not None:
                model_obs = (self.obs_normalizer.normalize_obs(obs)
                             if self.obs_normalizer is not None else obs)
                action, _ = self.model.predict(model_obs, deterministic=True)
                a = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))
                new_timeout = IDLE_MIN + (a + 1.0) * 0.5 * span
                self._apply_idle_timeout(new_timeout)

            # ------------- Reset per-minute counters -------------
            self.minute_arrivals_count = 0
            self.minute_cold_start_count = 0
            self.minute_blocked_count = 0

            self._minute_level_time = defaultdict(float)
            self._last_active_t = minute_start

            # ------------- Inject trace arrivals -------------
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

            # ------------- Build next observation -------------
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

    # ----------------------- idle-timeout application ------------------------
    def _apply_idle_timeout(self, timeout: float):
        """Apply a new idle-timeout setpoint. ``self.idle_timeout`` governs every
        container spawned from now on (upcoming pods, read by the base System on
        spawn). When ``self.retime_idle`` is set (ANALYSIS + APPLY_TO_IDLE) we
        also retime the *currently* idle containers so the live pool tracks the
        setpoint immediately; RL leaves this off to match its training."""
        self.idle_timeout = timeout
        if self.retime_idle:
            self._retime_idle_pool(timeout)

    def _retime_idle_pool(self, timeout: float):
        """Re-point every live container's per-instance setpoint to ``timeout``
        and move each currently-idle container's eviction deadline to
        idle_since + timeout (evicting immediately if it has already passed)."""
        now = self.env.now
        for server in self.servers:
            for c in server.containers:
                c.idle_timeout = timeout
        for c in list(self.idle_containers):
            if c.state != "Idle" or c.idle_since < 0:
                continue
            c.cancel_idle_timer()
            delay = max(0.0, c.idle_since + timeout - now)
            c.idle_timer_proc = self.env.process(_evict_after(self, c, delay))

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
        """Detect block / cold-start decision at entry time, then delegate."""
        if not self.trace_driven:
            self.minute_arrivals_count += 1
        chosen = next((c for c in self.idle_containers if c.state == "Idle"), None)
        if chosen is None:
            if self.find_server_for_spawn(request.resource_info) is None:
                self.minute_blocked_count += 1
            else:
                self.minute_cold_start_count += 1
        yield from super().handle_request(request)


# ----------------------------- controller ------------------------------------
class AnalysisController:
    """Analytical policy that duck-types the RL ``predict(obs) -> (action, _)``
    interface so ``LoggingSystem.minute_ticker`` can drive it interchangeably
    with a trained RL model.

    ``predict(obs)`` is called once at the start of each minute with
    obs = [arrivals_last_min, warm_pool, cold_hits, ram_util]. This class
    forecasts next-minute lambda from the obs/trace, solves for the SLA-optimal
    idle timeout analytically, and returns it encoded as an action in [-1, 1].
    Applying the timeout to the live pool is handled centrally by
    ``LoggingSystem._apply_idle_timeout`` (shared with the RL path).
    """

    def __init__(self, params: dict, predictor: str, ewma_alpha: float,
                 trace=None):
        if predictor not in ("ml", "last", "ewma", "oracle"):
            raise ValueError(f"Unknown PREDICTOR {predictor!r}.")
        if predictor in ("ml", "oracle") and trace is None:
            raise ValueError(f"PREDICTOR={predictor!r} needs the arrival trace.")
        self.params = params
        self.predictor = predictor
        self.ewma_alpha = ewma_alpha
        self.trace = trace
        self.minute = 0
        self.ewma = None
        if predictor == "ml":
            self._ml_predict, self._ml_lookback, _ = _get_ml_predictor()

    # -- lambda forecast (req/s) for the coming minute -------------------------
    def _forecast_rate(self, arrivals_last_min: float) -> float:
        if self.predictor == "oracle":
            return max(0.0, float(self.trace[self.minute])) / SECONDS_PER_MINUTE
        if self.predictor == "ml":
            m = self.minute
            if m >= self._ml_lookback:
                # Same window convention as multi_ML_proactive.minute_ticker.
                window = self.trace[m - self._ml_lookback:m]
                return max(0.0, float(self._ml_predict(window))) / SECONDS_PER_MINUTE
            # Predictor warm-up: fall back to the last observed minute.
            return (arrivals_last_min / SECONDS_PER_MINUTE) if m > 0 else 0.0
        # The minute-0 observation is the all-zeros bootstrap obs, not a real
        # measurement: skip it (the controller starts at IDLE_MIN and corrects
        # itself from minute 1 on).
        if self.minute == 0:
            return 0.0
        if self.predictor == "last":
            return arrivals_last_min / SECONDS_PER_MINUTE
        if self.ewma is None:
            self.ewma = arrivals_last_min
        else:
            self.ewma = (self.ewma_alpha * arrivals_last_min
                         + (1.0 - self.ewma_alpha) * self.ewma)
        return self.ewma / SECONDS_PER_MINUTE

    # -- policy interface -------------------------------------------------------
    def predict(self, obs, deterministic=True):
        lam = self._forecast_rate(float(np.asarray(obs).flatten()[0]))
        timeout = solve_optimal_timeout(
            lam, self.params["e_bw"], self.params["e_bc"],
            self.params["r_t"], self.params["r_a"], self.params["r_w"],
            LATENCY_SLA, IDLE_MIN, IDLE_MAX, TIMEOUT_GRID_STEP,
        )
        self.minute += 1
        # Inverse of minute_ticker's action -> timeout mapping.
        a = 2.0 * (timeout - IDLE_MIN) / (IDLE_MAX - IDLE_MIN) - 1.0
        return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32), None


def _evict_after(system, container, delay):
    """Deadline-based replacement for Container._idle_timeout_process: evict
    ``container`` after ``delay`` seconds unless it was reused meanwhile."""
    try:
        yield system.env.timeout(delay)
        if container.state != "Idle":
            return
        if container in system.idle_containers:
            system.update_idle_stats()
            system.idle_containers.remove(container)
        container.idle_timer_proc = None
        container.release_resources()
    except simpy.Interrupt:
        return


# ----------------------------- runner ----------------------------------------
def run_one_repetition(rep_idx: int, seed: int, cfg: dict, trace,
                       n_minutes: int, params: dict):
    global _ACTIVE_TRACKER
    random.seed(seed)
    np.random.seed(seed)
    Container.id_counter = itertools.count()

    env = simpy.Environment()

    # Select the per-minute policy by control mode:
    #   STATIC    (USE_ML False)        -> no policy; config idle timeout stays fixed.
    #   RL        (USE_ML, USE_RL)      -> trained checkpoint + its VecNormalize.
    #   ANALYSIS  (USE_ML, not USE_RL)  -> analytical controller.
    if not USE_ML:
        model, obs_normalizer = None, None
    elif USE_RL:
        model, obs_normalizer = _get_rl_policy()
    else:
        model = AnalysisController(
            params, PREDICTOR, EWMA_ALPHA,
            trace=trace if PREDICTOR in ("ml", "oracle") else None,
        )
        obs_normalizer = None

    # Retime the live pool only in ANALYSIS mode (when APPLY_TO_IDLE); RL must
    # affect upcoming containers only, matching how the policy was trained.
    retime_idle = USE_ML and (not USE_RL) and APPLY_TO_IDLE

    system = LoggingSystem(
        env, cfg,
        distribution=cfg["distribution"],
        trace_per_minute=trace,
        use_ml=USE_ML,                # gates the per-minute predict() call
        model=model,
        obs_normalizer=obs_normalizer,
        repetition=rep_idx,
        seed=seed,
        n_minutes=n_minutes,
        verbose=cfg["system"]["verbose"],
        retime_idle=retime_idle,
    )
    _ACTIVE_TRACKER = system

    for i in range(cfg["system"]["num_servers"]):
        system.add_server(Server(env, f"Server-{i}", cfg["server"]))

    env.process(system.minute_ticker())
    env.process(system.request_generator())
    env.process(system.warmup_reset_process())
    env.run(until=cfg["system"]["sim_time"] + 1.0)

    p95_lat, p99_lat = system.get_latency_percentiles()
    for row in system.minute_log:
        row["p95_latency"] = p95_lat
        row["p99_latency"] = p99_lat
    return system


def format_run_summary(system: LoggingSystem) -> str:
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


def write_log(all_rows, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return csv_path


# ----------------------------- parallel execution ----------------------------
def _worker_init():
    _install_container_hooks()
    # Both the RL policy and the analytical-ML forecaster pull in torch; cap its
    # thread pool so the worker processes don't oversubscribe the CPU.
    if USE_RL or PREDICTOR == "ml":
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass


def _rep_worker(task):
    rep_idx, seed, cfg, trace, n_minutes, params = task
    system = run_one_repetition(rep_idx, seed, cfg, trace, n_minutes, params)
    return rep_idx, system.minute_log, format_run_summary(system)


def _run_repetitions_parallel(tasks):
    n_workers = NUM_WORKERS or min(len(tasks), os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(tasks)))
    if n_workers == 1:
        _worker_init()
        return [_rep_worker(t) for t in tasks]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        return pool.map(_rep_worker, tasks)


def main():
    _install_container_hooks()

    n_minutes = DAYS_TO_RUN * MINUTES_PER_DAY
    trace = load_trace(TRACE_CSV, n_minutes)
    sim_time = n_minutes * SECONDS_PER_MINUTE

    # Resolve the control mode and the per-mode setup (warmup, params, checks).
    #   static   -> fixed config idle timeout, no policy, no queueing params.
    #   rl       -> trained checkpoint; validate the files exist up front.
    #   analysis -> analytical controller; ML predictor sets warmup to lookback.
    params = None
    warmup_minutes = WARMUP_MINUTES
    mode_note = ""

    if not USE_ML:
        mode = "static"
        mode_note = (f"fixed idle timeout {STATIC_IDLE_TIMEOUT}s "
                     f"(variables.py idle_cpu_timeout)")
    elif USE_RL:
        mode = "rl"
        if ALGORITHM.upper() not in {"PPO", "SAC"}:
            raise ValueError(f"Unknown ALGORITHM {ALGORITHM!r}; choose 'PPO' or 'SAC'.")
        if not Path(f"{RL_MODEL_PATH}.zip").exists():
            raise FileNotFoundError(
                f"RL checkpoint not found: {RL_MODEL_PATH}.zip\n"
                f"Train it with dynamic_pool/RL/train.py (ALGORITHM={ALGORITHM}, "
                f"TRACE={TRACE_NAME})."
            )
        if not RL_VECNORM_PATH.exists():
            raise FileNotFoundError(f"VecNormalize stats not found: {RL_VECNORM_PATH}")
        mode_note = (f"{ALGORITHM.upper()} {'best' if USE_BEST else 'final'} "
                     f"checkpoint ({RL_MODEL_PATH.parent.name})")
    else:
        mode = "analysis"
        if PREDICTOR == "ml":
            _, lookback, model_type = _get_ml_predictor()
            warmup_minutes = lookback
            mode_note = f"analytical, PREDICTOR=ml ({model_type}, lookback={lookback} min)"
        else:
            mode_note = (f"analytical, PREDICTOR={PREDICTOR}"
                         + (f" (alpha={EWMA_ALPHA})" if PREDICTOR == "ewma" else ""))

    cfg = build_config(sim_time, warmup_minutes=warmup_minutes)
    if mode == "analysis":
        params = resolve_model_params(cfg)

    print("\n=== dynamic_pool/multi_dynamic.py — idle-timeout control ===")
    print(f"  Mode:             {mode.upper()}  ({mode_note})")
    print(f"  USE_ML / USE_RL:  {USE_ML} / {USE_RL}")
    print(f"  Trace:            {TRACE_NAME} ({DAYS_TO_RUN} days, {n_minutes} minutes)")
    print(f"  Trace mean:       {trace.mean():.2f} req/min "
          f"(min={trace.min():.0f}, max={trace.max():.0f})")
    print(f"  NUM_REPETITIONS:  {NUM_REPETITIONS}  (BASE_SEED={BASE_SEED})")
    if mode != "static":
        print(f"  Timeout range:    [{IDLE_MIN}, {IDLE_MAX}]s")
    if mode == "analysis":
        print(f"  APPLY_TO_IDLE:    {APPLY_TO_IDLE} (retime current idle pool)")
    elif mode == "rl":
        print(f"  Retime existing:  False (upcoming containers only)")
    print(f"  Warmup:           {warmup_minutes} min discarded")
    if mode == "analysis":
        p_min = 1.0 - (LATENCY_SLA - params["e_bw"]) / params["e_bc"]
        print(f"  LATENCY_SLA:      {LATENCY_SLA}s  ->  required p_w >= {p_min:.3f}")
        print(f"  RESOURCE:         {RESOURCE}  "
              f"(r_t={params['r_t']}, r_a={params['r_a']}, r_w={params['r_w']})")
        print(f"  E[Bw]/E[Bc]:      {params['e_bw']:.3f}s / {params['e_bc']:.3f}s  "
              f"(E[Br] range [{params['e_bw']:.2f}, "
              f"{expected_latency(0.0, params['e_bw'], params['e_bc']):.2f}]s)")
        print(f"  Timeout grid:     {TIMEOUT_GRID_STEP}s")
    print(f"  Workers:          {NUM_WORKERS or 'auto'}")

    print("\n--- Repetitions ---")
    tasks = [(rep, BASE_SEED + rep, cfg, trace, n_minutes, params)
             for rep in range(NUM_REPETITIONS)]
    results = _run_repetitions_parallel(tasks)

    all_rows = []
    for _rep_idx, minute_log, summary in results:
        all_rows.extend(minute_log)
        print(summary)

    ts = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    csv_path = write_log(
        all_rows, LOG_DIR / f"{ts}_multi_x{NUM_REPETITIONS}_{mode}.csv")
    print(f"\nCombined per-minute log ({len(all_rows)} rows) written to:")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
