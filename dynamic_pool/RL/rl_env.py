"""Gymnasium environment that wraps the dynamic-pool serverless simulator.

State  : [arrivals, warm_pool_size, cold_hits, ram_utilization]  (last step)
Action : continuous in [-1, 1] -> scaled to [idle_timeout_min, idle_timeout_max]
Reward : -cold_start_ratio - ram_penalty * ram_utilization

Each step advances the SimPy simulation by ``step_duration`` seconds, after
which the agent observes the deltas and is asked for a new idle timeout.

Two arrival modes:
  - "synthetic": uses the System's built-in request_generator (config-driven).
  - "trace":     per-minute Poisson arrivals from a real-trace ndarray. Each
                 reset() picks a random window of ``max_steps * step_duration``
                 sim-seconds from the trace (or starts at 0 if ``random_window``
                 is False).
"""

from __future__ import annotations

import copy
import os
import random
import sys

import gymnasium as gym
import numpy as np
import simpy
from gymnasium import spaces

_RL_DIR = os.path.dirname(os.path.abspath(__file__))
_DYNAMIC_DIR = os.path.dirname(_RL_DIR)
_REPO_ROOT = os.path.dirname(_DYNAMIC_DIR)
for p in (_REPO_ROOT, _DYNAMIC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from Server import Server
from Request import Request
from dynamic_pool.System import System


SECONDS_PER_MINUTE = 60


class DynamicPoolEnv(gym.Env):
    """RL env controlling the idle timeout of newly-spawned containers."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_config: dict,
        step_duration: float = 60.0,
        max_steps: int = 200,
        idle_timeout_min: float = 1.0,
        idle_timeout_max: float = 600.0,
        ram_penalty: float = 0.5,
        seed: int | None = None,
        arrival_mode: str = "synthetic",
        trace: np.ndarray | None = None,
        random_window: bool = True,
    ):
        super().__init__()
        self.base_config = copy.deepcopy(base_config)
        self.step_duration = float(step_duration)
        self.max_steps = int(max_steps)
        self.idle_timeout_min = float(idle_timeout_min)
        self.idle_timeout_max = float(idle_timeout_max)
        self.ram_penalty = float(ram_penalty)
        self.arrival_mode = arrival_mode
        self.random_window = bool(random_window)

        if arrival_mode == "trace":
            if trace is None:
                raise ValueError("arrival_mode='trace' requires a `trace` ndarray.")
            self.trace = np.asarray(trace, dtype=float)
            # How many trace-minutes are consumed in one episode.
            self.episode_minutes = int(
                np.ceil(self.max_steps * self.step_duration / SECONDS_PER_MINUTE)
            )
            if len(self.trace) < self.episode_minutes:
                raise ValueError(
                    f"Trace has {len(self.trace)} minutes; need at least "
                    f"{self.episode_minutes} for one episode."
                )
        elif arrival_mode != "synthetic":
            raise ValueError(f"Unknown arrival_mode: {arrival_mode!r}")

        # RL trains over the full episode — no separate warm-up exclusion
        self.base_config["system"]["warmup_time"] = 0
        self.base_config["system"]["verbose"] = False

        # [arrivals, warm_pool_size, cold_hits, ram_utilization] over the step.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._seed = seed
        self._env = None
        self._system = None
        self._current_step = 0
        self._prev_gen = 0
        self._prev_cold = 0
        self._prev_ram_area = 0.0

    # ------------------------------------------------------------------

    def _trace_arrival_process(self, window):
        """Per-minute Poisson arrivals from the trace window."""
        env = self._env
        system = self._system
        for count in window:
            minute_start = env.now
            elapsed = 0.0
            if count > 0:
                rate_per_second = count / SECONDS_PER_MINUTE
                while True:
                    gap = random.expovariate(rate_per_second)
                    if elapsed + gap >= SECONDS_PER_MINUTE:
                        break
                    yield env.timeout(gap)
                    elapsed += gap
                    request = Request(next(system.req_id_counter), env.now,
                                      system.config["request"])
                    if system.service_distribution == "traces":
                        request.service_time = system.get_next_trace_service_time()
                    system.request_stats["generated"] += 1
                    env.process(system.handle_request(request))
            remaining = SECONDS_PER_MINUTE - (env.now - minute_start)
            if remaining > 0:
                yield env.timeout(remaining)

    # ------------------------------------------------------------------

    def _build(self, seed: int | None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._env = simpy.Environment()
        self._system = System(
            self._env,
            self.base_config,
            distribution=self.base_config["distribution"],
            verbose=False,
        )
        for i in range(self.base_config["system"]["num_servers"]):
            self._system.add_server(
                Server(self._env, f"Server-{i}", self.base_config["server"])
            )

        if self.arrival_mode == "synthetic":
            self._env.process(self._system.request_generator())
        else:
            if self.random_window:
                max_start = len(self.trace) - self.episode_minutes
                start = random.randint(0, max_start) if max_start > 0 else 0
            else:
                start = 0
            window = self.trace[start:start + self.episode_minutes]
            self._env.process(self._trace_arrival_process(window))

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._seed
        if episode_seed is None:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._build(episode_seed)

        self._current_step = 0
        self._prev_gen = 0
        self._prev_cold = 0
        self._system.update_resource_stats()
        self._prev_ram_area = self._system.total_ram_usage_area

        obs = np.zeros(4, dtype=np.float32)
        return obs, {"episode_seed": episode_seed}

    # ------------------------------------------------------------------

    def step(self, action):
        a = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))
        span = self.idle_timeout_max - self.idle_timeout_min
        new_timeout = self.idle_timeout_min + (a + 1.0) * 0.5 * span
        # Affects all containers created from now on (System reads this on spawn).
        self._system.idle_timeout = new_timeout

        self._env.run(until=self._env.now + self.step_duration)

        # Deltas
        gen_now = self._system.request_stats["generated"]
        cold_now = self._system.request_stats["container_spawns_initiated"]
        self._system.update_resource_stats()
        ram_area_now = self._system.total_ram_usage_area

        d_gen = gen_now - self._prev_gen
        d_cold = cold_now - self._prev_cold
        d_ram_area = ram_area_now - self._prev_ram_area

        self._prev_gen = gen_now
        self._prev_cold = cold_now
        self._prev_ram_area = ram_area_now

        cold_ratio = d_cold / d_gen if d_gen > 0 else 0.0
        total_cap = self._system.total_ram_capacity
        ram_util = (d_ram_area / (self.step_duration * total_cap)) if total_cap > 0 else 0.0

        reward = -cold_ratio - self.ram_penalty * ram_util

        obs = np.array(
            [d_gen, len(self._system.idle_containers), d_cold, ram_util],
            dtype=np.float32,
        )

        self._current_step += 1
        truncated = self._current_step >= self.max_steps
        terminated = False

        info = {
            "idle_timeout": new_timeout,
            "cold_ratio": cold_ratio,
            "ram_util": ram_util,
            "d_gen": d_gen,
            "d_cold": d_cold,
            "reward_components": {
                "cold_ratio": cold_ratio,
                "ram_util": ram_util,
            },
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------

    def close(self):
        self._env = None
        self._system = None
