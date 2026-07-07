"""ENSURE-style orchestrator on top of ``dynamic_pool.System``.

Implements the two scheduler ideas from the ENSURE paper that translate to a
reserved-capacity, single-application simulator:

  * **FnSched (packing)** — when reusing a warm container, pick the one on the
    lowest-indexed server. ``find_server_for_spawn`` already iterates servers
    in index order, so cold spawns are first-fit-packed too. Result: load is
    concentrated on a few servers; high-indexed servers stay idle and time out.

  * **FnScale (square-root staffing)** — keep ``ceil(sqrt(R))`` protected warm
    containers in the pool, where R is the count of containers "active in the
    last 5 s" (currently serving plus recently released). The buffer absorbs
    bursts before they trigger user-visible cold starts.

The ENSURE per-class cpu-shares regulation and ET/MP classifier do not map to
this simulator (single app, reserved CPU slots) and are intentionally omitted.
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import defaultdict

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import simpy

from Request import Request
from dynamic_pool.System import System, _sample_truncated_cauchy
from ensure.Container import EnsureContainer


SECONDS_PER_MINUTE = 60


class EnsureSystem(System):
    def __init__(self, env, config, distribution, trace_per_minute=None,
                 n_minutes=None, repetition=0, seed=0, verbose=False):
        super().__init__(env, config, distribution, verbose=verbose)
        ensure_cfg = config.get("ensure", {})
        self.active_window = float(ensure_cfg.get("active_window", 5.0))
        self.scale_check_interval = float(ensure_cfg.get("scale_check_interval", 5.0))
        self.sqrt_c = float(ensure_cfg.get("sqrt_staffing_c", 1.0))

        if trace_per_minute is not None:
            import numpy as np
            self.trace = np.asarray(trace_per_minute, dtype=float)
            self.trace_driven = True
            self.n_minutes = len(self.trace)
        else:
            self.trace = None
            self.trace_driven = False
            self.n_minutes = int(n_minutes) if n_minutes is not None else 0

        self.repetition = repetition
        self.seed = seed

        # Per-minute counters (for logging)
        self.minute_arrivals_count = 0
        self.minute_cold_start_count = 0
        self.minute_blocked_count = 0
        self.minute_reuse_count = 0
        self.minute_log = []

        # FnScale buffer bookkeeping
        self._buffer_spawns_inflight = 0

    # ------------------------------------------------------------------
    # FnSched: server index lookup
    # ------------------------------------------------------------------
    def _server_index(self, server):
        try:
            return self.servers.index(server)
        except ValueError:
            return len(self.servers)

    # ------------------------------------------------------------------
    # Arrival generator — trace mode emits arrivals from the minute ticker.
    # ------------------------------------------------------------------
    def request_generator(self):
        if not self.trace_driven:
            yield from super().request_generator()
        return

    def minute_ticker(self):
        """Per-minute trace injection + logging. Trace-driven mode only."""
        if not self.trace_driven:
            return
        for m in range(self.n_minutes):
            self.minute_arrivals_count = 0
            self.minute_cold_start_count = 0
            self.minute_blocked_count = 0
            self.minute_reuse_count = 0

            rate = max(0.0, float(self.trace[m]))
            if rate > 0:
                self.env.process(self._inject_arrivals_exponential(rate))

            yield self.env.timeout(SECONDS_PER_MINUTE)

            self.update_resource_stats()
            mean_lat = (self.latency_stats['total_latency'] / self.latency_stats['count']
                        if self.latency_stats['count'] > 0 else 0.0)
            R = self._compute_R()
            buffer_target = math.ceil(self.sqrt_c * math.sqrt(R)) if R > 0 else 0
            protected_count = sum(1 for c in self.idle_containers
                                  if getattr(c, "protected", False))
            self.minute_log.append({
                "repetition": self.repetition,
                "seed": self.seed,
                "minute": m,
                "time": self.env.now,
                "arrival": self.minute_arrivals_count,
                "accepted": self.minute_arrivals_count - self.minute_blocked_count,
                "blocked": self.minute_blocked_count,
                "cold_hit": self.minute_cold_start_count,
                "reuse": self.minute_reuse_count,
                "warm_pool_size": len(self.idle_containers),
                "ensure_R": R,
                "ensure_buffer_target": buffer_target,
                "ensure_buffer_active": protected_count,
                "energy": self.total_energy_usage,
                "ram_area": self.total_ram_usage_area,
                "cpu_area": self.total_cpu_usage_area,
                "mean_latency": mean_lat,
            })

    def _inject_arrivals_exponential(self, count):
        rate_per_second = count / SECONDS_PER_MINUTE
        if rate_per_second <= 0:
            return
        elapsed = 0.0
        while True:
            gap = random.expovariate(rate_per_second)
            if elapsed + gap >= SECONDS_PER_MINUTE:
                return
            yield self.env.timeout(gap)
            elapsed += gap
            request = Request(next(self.req_id_counter), self.env.now,
                              self.config["request"])
            if self.service_distribution == "traces":
                request.service_time = self.get_next_trace_service_time()
            self.minute_arrivals_count += 1
            if self.env.now >= self.warmup_time:
                self.request_stats["generated"] += 1
            self.env.process(self.handle_request(request))

    # ------------------------------------------------------------------
    # FnSched: pack reuse onto lowest-indexed server.
    # ------------------------------------------------------------------
    def handle_request(self, request):
        start_time = self.env.now
        self.increment_waiting()

        # Reuse: prefer protected warm containers, then by lowest server index.
        idle_candidates = [c for c in self.idle_containers if c.state == "Idle"]
        if idle_candidates:
            chosen = min(
                idle_candidates,
                key=lambda c: (
                    0 if getattr(c, "protected", False) else 1,
                    self._server_index(c.server),
                    c.id,
                ),
            )
            chosen.state = "Pending"
            self.update_idle_stats()
            self.idle_containers.remove(chosen)
            chosen.cancel_idle_timer()
            # Promoted out of the protected buffer — it's now serving traffic.
            if getattr(chosen, "protected", False):
                chosen.protected = False
            if start_time >= self.warmup_time:
                self.request_stats['container_reuses'] += 1
                self.minute_reuse_count += 1
            self.env.process(self.container_service_lifecycle(request, chosen, start_time))
            return

        # No warm container — cold spawn on lowest-indexed server with room.
        server = self.find_server_for_spawn(request.resource_info)
        if server:
            if start_time >= self.warmup_time:
                self.request_stats['container_spawns_initiated'] += 1
                self.minute_cold_start_count += 1
            self.allocate_server(server, request)
            self.increment_cold_starting()
            spawned = yield self.env.process(self.spawn_container_process(server, request))
            self.decrement_cold_starting()
            if not spawned:
                raise RuntimeError(f"{self.env.now:.2f} - Container spawn failed on {server}.")
            if start_time >= self.warmup_time:
                self.request_stats['container_spawns_succeeded'] += 1
            self.env.process(self.container_service_lifecycle(request, spawned, start_time))
        else:
            if start_time >= self.warmup_time:
                self.request_stats['blocked_no_server_capacity'] += 1
                self.minute_blocked_count += 1
            request.state = "Rejected"
            self.decrement_waiting()

    # ------------------------------------------------------------------
    # Override the spawn process so on-demand containers are EnsureContainers
    # (so protected=False explicitly, and the warm pool is homogeneous).
    # ------------------------------------------------------------------
    def spawn_container_process(self, server, request):
        try:
            if (server.cpu_real < 0 or server.ram_real < 0 or
                    server.cpu_reserve < 0 or server.ram_reserve < 0):
                raise RuntimeError(
                    f"{self.env.now:.2f} - Resource allocation failed on {server} for {request}."
                )

            spawn_time = self._sample_spawn_time()
            yield self.env.timeout(spawn_time)

            container = EnsureContainer(
                self.env, self, server, request.resource_info,
                idle_timeout=self.idle_timeout, is_cold_start=True,
                protected=False,
            )
            server.containers.append(container)
            request.spawn_time += spawn_time
            return container
        except Exception as e:
            print(f"ERROR: {self.env.now:.2f} - spawn failed for {request} on {server}: {e}")
            raise

    def _sample_spawn_time(self):
        """Mirror the base System.spawn_container_process distribution switch.

        Note: variables.py uses ``spawn-distribution: "traces"`` (plural), but
        the base only treats ``"trace"`` (singular) as Cauchy-sampled — the
        plural form falls through to the exponential default. We preserve that
        behavior to stay consistent with the existing dynamic-pool runs.
        """
        spawn_distribution = self.distribution["spawn-distribution"]
        if spawn_distribution == "deterministic":
            return self.spawn_time_mean
        if spawn_distribution == "lognormal":
            return random.lognormvariate(self.spawn_lognormal_mu, self.spawn_lognormal_sigma)
        if spawn_distribution == "trace" and self._spawn_cauchy is not None:
            return _sample_truncated_cauchy(*self._spawn_cauchy)
        return random.expovariate(1.0 / self.spawn_time_mean)

    # ------------------------------------------------------------------
    # FnScale: maintain ceil(sqrt(R)) protected warm containers.
    # ------------------------------------------------------------------
    def _compute_R(self):
        """R = containers active in the last ``active_window`` seconds.

        Proxy: currently-serving + idle containers whose ``idle_since`` is
        within the window (they just released a request).
        """
        now = self.env.now
        idle_recent = sum(
            1 for c in self.idle_containers
            if not getattr(c, "protected", False)
            and c.idle_since >= 0
            and (now - c.idle_since) <= self.active_window
        )
        return self.request_running + idle_recent

    # Hard cap on protected spawns scheduled per fnscale tick — prevents
    # runaway queueing if R or sqrt(R) ever spikes pathologically.
    _FNSCALE_MAX_PER_TICK = 32

    def fnscale_process(self):
        """Periodically top up the protected √R warm buffer.

        Resource deductions happen synchronously inside this loop (not deferred
        into the spawned coroutine) so successive ``find_server_for_spawn``
        calls see updated capacity and naturally roll over to the next server
        instead of all targeting the same one.
        """
        ri = self.config["request"]
        while True:
            yield self.env.timeout(self.scale_check_interval)
            R = self._compute_R()
            target = math.ceil(self.sqrt_c * math.sqrt(R)) if R > 0 else 0
            protected_now = sum(1 for c in self.idle_containers
                                if getattr(c, "protected", False))
            deficit = target - (protected_now + self._buffer_spawns_inflight)
            deficit = min(deficit, self._FNSCALE_MAX_PER_TICK)
            for _ in range(deficit):
                server = self.find_server_for_spawn(ri)
                if server is None:
                    break
                # Deduct synchronously: next find_server_for_spawn sees this.
                self.update_resource_stats()
                server.cpu_real -= ri['cold_start_cpu']
                server.ram_real -= ri['cold_start_ram']
                server.cpu_reserve -= ri['cpu_demand']
                server.ram_reserve -= ri['ram_demand']
                self._buffer_spawns_inflight += 1
                self.env.process(self._spawn_protected_warm_finalize(server))

    def _spawn_protected_warm_finalize(self, server):
        """Finish a protected-warm spawn whose resources were pre-deducted.

        Just waits out the spawn delay, creates the container, and transitions
        it from cold-start to warm idle (handing the cold-start surplus back).
        """
        resource_info = self.config["request"]
        try:
            self.increment_cold_starting()
            spawn_time = self._sample_spawn_time()
            yield self.env.timeout(spawn_time)
            self.decrement_cold_starting()

            container = EnsureContainer(
                self.env, self, server, resource_info,
                idle_timeout=self.idle_timeout, is_cold_start=True,
                protected=True,
            )
            server.containers.append(container)

            # Cold -> warm idle: hand back the cold-start surplus.
            self.update_resource_stats()
            delta_cpu = resource_info['cold_start_cpu'] - resource_info['warm_cpu']
            delta_ram = resource_info['cold_start_ram'] - resource_info['warm_ram']
            server.cpu_real += delta_cpu
            server.ram_real += delta_ram
            container.cpu_current = resource_info['warm_cpu']
            container.ram_current = resource_info['warm_ram']
            container.state = "Idle"
            container.idle_since = self.env.now

            self.update_idle_stats()
            self.idle_containers.append(container)
            # No idle timer because protected.
        finally:
            self._buffer_spawns_inflight -= 1
