import simpy
import random
import itertools
import math
import os
import csv

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from variables import *
from Request import Request
from fixed_pool.Container import Container
from Helper import weibull_params_from_mean_std, lognormal_params_from_mean_std


def _load_processing_profile(profile_path):
    """Read measured_traces/profile.csv into {type: (mean, std, lo, hi)}."""
    profile = {}
    with open(profile_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("type") or "").strip()
            if not t:
                continue
            try:
                profile[t] = (
                    float(row["mean"]), float(row["std"]),
                    float(row["min"]), float(row["max"]),
                )
            except (KeyError, ValueError, TypeError):
                continue
    return profile


def _sample_truncated_cauchy(loc, scale, lo, hi, max_iter=64):
    """Cauchy(loc, scale) sample rejection-truncated to [lo, hi]."""
    for _ in range(max_iter):
        x = loc + scale * math.tan(math.pi * (random.random() - 0.5))
        if lo <= x <= hi:
            return x
    return min(max(loc, lo), hi)


class System:
    """Orchestrates the simulation: manages servers, warm pool, and request lifecycle."""

    def __init__(self, env, config, distribution, verbose=False):
        self.env = env
        self.servers = []
        self.idle_containers = []       # Warm-pool containers available for immediate reuse
        self.config = config
        self.max_warm_queue = self.calculate_warm_number(config)
        print(f"Sim: Warm queue size: {self.max_warm_queue}")
        print(f"Sim: Simulation time: {config['system']['sim_time']}s")
        self.warmup_time = config["system"].get("warmup_time", 0)
        # Opt-in detailed record collection (per-request records + event-driven
        # resource snapshots). Off by default for performance; enabled by
        # validation scripts that need compute_batch_means(). See handle_request,
        # container_service_lifecycle, and update_resource_stats.
        self.collect_records = config["system"].get("collect_records", False)

        self.arrival_rate_mean = config["request"]["arrival_rate_mean"]
        self.arrival_rate_std = config["request"]["arrival_rate_std"]
        self.service_rate = config["request"]["service_rate"]
        self.spawn_time_mean = config["container"]["spawn_time_mean"]
        self.spawn_time_std = config["container"]["spawn_time_std"]

        self.distribution = distribution
        self.service_distribution = self.distribution.get("service-distribution", "exponential")
        self.service_trace_times = []
        self.service_trace_index = 0
        self.verbose = verbose
        self.req_id_counter = itertools.count()

        # Pre-calculate distribution parameters
        if self.distribution["arrival-distribution"] == "weibull":
            # Convert arrival rate (req/s) to inter-arrival time using the delta method
            inter_arrival_mean = 1.0 / self.arrival_rate_mean
            inter_arrival_std = self.arrival_rate_std / (self.arrival_rate_mean ** 2)
            self.arrival_weibull_shape, self.arrival_weibull_scale = weibull_params_from_mean_std(
                inter_arrival_mean, inter_arrival_std
            )
        if self.distribution["spawn-distribution"] == "lognormal":
            self.spawn_lognormal_mu, self.spawn_lognormal_sigma = lognormal_params_from_mean_std(
                self.spawn_time_mean, self.spawn_time_std
            )
        # Cauchy params loaded on demand from measured_traces/profile.csv.
        self._service_cauchy = None
        self._spawn_cauchy = None
        if (self.service_distribution == "traces"
                or self.distribution.get("spawn-distribution") == "trace"):
            self.load_service_traces()

        # Waiting-request tracking (for Little's Law)
        self.waiting_requests_count = 0
        self.last_waiting_update = 0.0
        self.total_waiting_area = 0.0

        # In-service request tracking
        self.request_running = 0
        self.last_processing_update = 0.0
        self.total_processing_area = 0.0

        # Resource usage tracking
        self.last_resource_update = 0.0
        self.total_cpu_usage_area = 0.0
        self.total_ram_usage_area = 0.0
        self.total_cpu_capacity = 0.0
        self.total_ram_capacity = 0.0
        self.total_energy_usage = 0.0

        # Warm/idle RAM tracking: time-integral of RAM held by warm-pool
        # containers while they sit Idle (RAM-seconds). Cold-start and
        # processing RAM are excluded — only containers in idle_containers with
        # state == "Idle" (which always hold warm_ram) are counted. Indicates
        # how much RAM is wasted while functions idle in the warm pool.
        self.last_idle_ram_update = 0.0
        self.total_idle_ram_area = 0.0

        # Per-request records and periodic resource snapshots
        self.request_records = []
        self.resource_history = []

        self.request_stats = {
            'generated': 0,
            'processed': 0,
            'blocked_no_server_capacity': 0,
            'container_spawns_initiated': 0,
            'container_spawns_succeeded': 0,
        }

        self.latency_stats = {
            'total_latency': 0.0,
            'spawning_time': 0.0,
            'processing_time': 0.0,
            'waiting_time': 0.0,
            'count': 0,
            'total_latency_sq': 0.0,   # For variance: E[B²]
            'all_latencies': [],        # Individual samples for percentile calculation
        }

    # ------------------------------------------------------------------
    # Trace loading
    # ------------------------------------------------------------------

    def load_service_traces(self):
        """Load Cauchy(loc, scale) params from measured_traces/profile.csv.

        Sets ``_service_cauchy`` (when service_distribution == "traces") and
        ``_spawn_cauchy`` (when spawn-distribution == "trace"). Also seeds
        ``service_trace_times`` with the trace mean so legacy callers that do
        ``np.mean(system.service_trace_times)`` still see the right value.
        """
        profile_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "traces", "measured_traces", "profile.csv",
        )
        try:
            profile = _load_processing_profile(profile_path)
        except FileNotFoundError:
            raise RuntimeError(f"Processing-time profile not found: {profile_path}")
        if self.service_distribution == "traces":
            if "service-time" not in profile:
                raise RuntimeError(f"'service-time' row missing in {profile_path}")
            self._service_cauchy = profile["service-time"]
            self.service_trace_times = [self._service_cauchy[0]]
            self.service_trace_index = 0
        if self.distribution.get("spawn-distribution") == "trace":
            if "cold-start" not in profile:
                raise RuntimeError(f"'cold-start' row missing in {profile_path}")
            self._spawn_cauchy = profile["cold-start"]

    def get_next_trace_service_time(self):
        """Sample a service time from the Cauchy fit of the measured traces."""
        if self._service_cauchy is None:
            raise RuntimeError("Service-time Cauchy params not loaded.")
        return _sample_truncated_cauchy(*self._service_cauchy)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def calculate_warm_number(self, config):
        """Compute the target warm-pool size from configuration."""
        resource = max(config["request"]["ram_demand"], config["request"]["cpu_demand"])
        max_containers = math.floor(100 / resource) * config["system"]["num_servers"]
        return int(max_containers * config["system"]["warm_percent"])

    def add_server(self, server):
        """Register a server with the system and update total capacity counters."""
        self.servers.append(server)
        self.total_cpu_capacity += server.cpu_capacity
        self.total_ram_capacity += server.ram_capacity

    # ------------------------------------------------------------------
    # Warmup reset
    # ------------------------------------------------------------------

    def warmup_reset_process(self):
        """Wait until warmup_time, then reset all time-weighted accumulators.

        This ensures that ``get_mean_*`` methods only reflect steady-state
        behaviour (post-warmup).
        """
        if self.warmup_time <= 0:
            return
        yield self.env.timeout(self.warmup_time)

        # Flush pending intervals into (now-discarded) accumulators
        self.update_waiting_stats()
        self.update_processing_stats()
        self.update_resource_stats()
        self.update_idle_ram_stats()

        # Reset time-weighted areas
        self.total_waiting_area = 0.0
        self.total_processing_area = 0.0
        self.total_cpu_usage_area = 0.0
        self.total_ram_usage_area = 0.0
        self.total_energy_usage = 0.0
        self.total_idle_ram_area = 0.0

        if self.verbose:
            print(f"{self.env.now:.2f} - Warmup complete. Accumulators reset for steady-state collection.")

    # ------------------------------------------------------------------
    # Pre-warming
    # ------------------------------------------------------------------

    def pre_warm(self):
        """Spawn the initial warm-pool containers before request generation begins."""
        print(f"Pre-warming {self.max_warm_queue} containers")
        request = Request(1111, 22222, self.config["request"])
        while len(self.idle_containers) < self.max_warm_queue:
            server = self.find_server_for_spawn(request.resource_info)
            if server is None:
                raise RuntimeError(
                    f"{self.env.now:.2f} - No server has enough resources for pre-warming."
                )
            if self.verbose:
                print(f"{self.env.now:.2f} - Spawning warm container on {server}.")
            self.request_stats['container_spawns_initiated'] += 1
            self.allocate_server(server, request, is_pre_warm=True)
            spawned = yield self.env.process(
                self.spawn_container_process(server, request, is_pre_warm=True)
            )
            if not spawned:
                raise RuntimeError(f"{self.env.now:.2f} - Container spawn failed on {server}.")
            self.update_idle_ram_stats()
            self.idle_containers.append(spawned)
        print(f"Pre-warming complete. {len(self.idle_containers)} containers are now idle.")

    # ------------------------------------------------------------------
    # Request generation
    # ------------------------------------------------------------------

    def request_generator(self):
        """Generate requests according to the configured arrival distribution."""
        while True:
            arrival_distribution = self.distribution["arrival-distribution"]
            if arrival_distribution == "exponential":
                inter_arrival_time = random.expovariate(self.arrival_rate_mean)
            elif arrival_distribution == "deterministic":
                inter_arrival_time = 1.0 / self.arrival_rate_mean
            elif arrival_distribution == "weibull":
                inter_arrival_time = random.weibullvariate(
                    self.arrival_weibull_scale, self.arrival_weibull_shape
                )
            else:
                inter_arrival_time = random.expovariate(self.arrival_rate_mean)

            yield self.env.timeout(inter_arrival_time)

            request = Request(next(self.req_id_counter), self.env.now, self.config["request"])
            if self.service_distribution == "traces":
                request.service_time = self.get_next_trace_service_time()
            if self.env.now >= self.warmup_time:
                self.request_stats['generated'] += 1
            if self.verbose:
                print(f"{self.env.now:.2f} - Request generated: {request}")
            self.env.process(self.handle_request(request))

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def handle_request(self, request):
        """Route an incoming request to a warm container or spawn a cold one."""
        start_time = self.env.now
        self.increment_waiting()

        # Try to reuse a warm-pool container
        chosen = next((c for c in self.idle_containers if c.state == "Idle"), None)
        if chosen:
            if self.verbose:
                print(f"{self.env.now:.2f} - Found idle container, assigning directly.")
            self.update_idle_ram_stats()   # flush before this pod leaves Idle
            chosen.state = "Pending"
            self.env.process(self.container_service_lifecycle(request, chosen, start_time))
            return

        # No warm container available — try spawning a cold one
        server = self.find_server_for_spawn(request.resource_info)
        if server:
            if self.verbose:
                print(f"{self.env.now:.2f} - No idle containers. Spawning cold container on {server}.")
            if start_time >= self.warmup_time:
                self.request_stats['container_spawns_initiated'] += 1
            self.allocate_server(server, request, is_pre_warm=False)
            spawned = yield self.env.process(self.spawn_container_process(server, request))
            if not spawned:
                raise RuntimeError(f"{self.env.now:.2f} - Container spawn failed on {server}.")
            if start_time >= self.warmup_time:
                self.request_stats['container_spawns_succeeded'] += 1
            self.env.process(self.container_service_lifecycle(request, spawned, start_time))
        else:
            if self.verbose:
                print(f"{self.env.now:.2f} - BLOCK: No server with sufficient capacity for {request}")
            if start_time >= self.warmup_time:
                self.request_stats['blocked_no_server_capacity'] += 1
            request.state = "Rejected"
            self.decrement_waiting()
            if self.collect_records:
                self.record_request_info(request)

    def find_server_for_spawn(self, resource_info):
        """Return the first server with sufficient capacity (First-Fit), or None."""
        for server in self.servers:
            if server.has_capacity(resource_info):
                return server
        return None

    def allocate_server(self, server, request, is_pre_warm=False):
        """Eagerly deduct required resources synchronously before yielding."""
        self.update_resource_stats()
        if is_pre_warm:
            server.cpu_real -= request.cpu_warm
            server.ram_real -= request.ram_warm
        else:
            server.cpu_real -= request.cpu_cold_start
            server.ram_real -= request.ram_cold_start
        server.cpu_reserve -= request.cpu_demand
        server.ram_reserve -= request.ram_demand

    def spawn_container_process(self, server, request, is_pre_warm=False):
        """Simulate the cold-start delay and reserve resources for a new container.

        Args:
            server:      Target server.
            request:     Triggering request (provides resource dimensions).
            is_pre_warm: True during pre-warming (uses warm-idle resource levels).
        """
        try:
            if (server.cpu_real < 0 or server.ram_real < 0 or
                    server.cpu_reserve < 0 or server.ram_reserve < 0):
                print(f"{self.env.now:.2f} - ERROR: Resource allocation failed on {server} for {request}.")
                exit(1)

            spawn_distribution = self.distribution["spawn-distribution"]
            if spawn_distribution == "exponential":
                spawn_time = random.expovariate(1.0 / self.spawn_time_mean)
            elif spawn_distribution == "deterministic":
                spawn_time = self.spawn_time_mean
            elif spawn_distribution == "lognormal":
                spawn_time = random.lognormvariate(
                    self.spawn_lognormal_mu, self.spawn_lognormal_sigma
                )
            elif spawn_distribution == "trace":
                if self._spawn_cauchy is None:
                    raise RuntimeError("Cold-start Cauchy params not loaded.")
                spawn_time = _sample_truncated_cauchy(*self._spawn_cauchy)
            else:
                spawn_time = random.expovariate(1.0 / self.spawn_time_mean)

            if self.verbose:
                print(f"{self.env.now:.2f} - Waiting {spawn_time:.2f}s for container spawn...")
            yield self.env.timeout(spawn_time)

            container = Container(
                self.env, self, server, request.resource_info, is_cold_start=not is_pre_warm
            )
            server.containers.append(container)
            if self.verbose:
                print(f"{self.env.now:.2f} - Spawn complete: {container}")
            request.spawn_time += spawn_time
            return container

        except Exception as e:
            print(f"ERROR: {self.env.now:.2f} - Unexpected error during spawn for {request} on {server}: {e}")
            self.env.exit(None)

    def container_service_lifecycle(self, request, container, start_time):
        """Assign request to container, wait for service completion, then release."""
        container.assign_request(request)

        total_wait = self.env.now - start_time
        request.wait_time = total_wait
        self.decrement_waiting()
        is_steady_state = start_time >= self.warmup_time
        # if is_steady_state:
        #     self.latency_stats['waiting_time'] += total_wait
        if self.verbose:
            print(f"{self.env.now:.2f} - {request} waited {total_wait:.2f}s")

        self.increment_processing()
        service_time = (
            request.service_time
            if self.service_distribution == "traces"
            else random.expovariate(self.service_rate)
        )
        if self.verbose:
            print(f"{self.env.now:.2f} - {request} starting service in {container}. "
                  f"Expected duration: {service_time:.2f}s")

        yield self.env.timeout(service_time)

        if is_steady_state:
            self.request_stats['processed'] += 1
        request.state = "Finished"
        container.release_request()
        self.decrement_processing()

        processing_time = request.end_service_time - request.start_service_time
        total_latency = request.wait_time + service_time
        if is_steady_state:
            self.latency_stats['total_latency'] += total_latency
            # self.latency_stats['total_latency_sq'] += total_latency ** 2
            self.latency_stats['all_latencies'].append(total_latency)
            # self.latency_stats['spawning_time'] += request.spawn_time
            # self.latency_stats['processing_time'] += processing_time
            self.latency_stats['count'] += 1

        if self.collect_records:
            self.record_request_info(request)

        if self.verbose:
            print(f"{self.env.now:.2f} - {request} latencies: Total {total_latency:.2f}s, "
                  f"Spawn {request.spawn_time:.2f}s, "
                  f"Wait {request.wait_time:.2f}s, Processing {processing_time:.2f}s")

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def update_waiting_stats(self):
        """Accumulate time-weighted waiting-request count."""
        now = self.env.now
        self.total_waiting_area += (now - self.last_waiting_update) * self.waiting_requests_count
        self.last_waiting_update = now

    def increment_waiting(self):
        self.update_waiting_stats()
        self.waiting_requests_count += 1

    def decrement_waiting(self):
        self.update_waiting_stats()
        self.waiting_requests_count -= 1

    def update_processing_stats(self):
        """Accumulate time-weighted in-service request count."""
        now = self.env.now
        self.total_processing_area += (now - self.last_processing_update) * self.request_running
        self.last_processing_update = now

    def increment_processing(self):
        self.update_processing_stats()
        self.request_running += 1

    def decrement_processing(self):
        self.update_processing_stats()
        self.request_running -= 1

    def update_resource_stats(self):
        """Accumulate time-weighted CPU/RAM usage and energy consumption.

        Called whenever resources change (spawn, assign, release).  Also
        records an event-driven snapshot in ``resource_history`` so that
        ``compute_batch_means`` can build precise time-weighted per-batch
        averages.
        """
        now = self.env.now
        dt = now - self.last_resource_update
        cpu_used = sum(s.cpu_capacity - s.cpu_real for s in self.servers)
        ram_used = sum(s.ram_capacity - s.ram_real for s in self.servers)
        power_used = sum(s.current_power() for s in self.servers)
        self.total_cpu_usage_area += dt * cpu_used
        self.total_ram_usage_area += dt * ram_used
        self.total_energy_usage += power_used * dt
        self.last_resource_update = now

        # Event-driven resource snapshot — only consumed by compute_batch_means().
        # Gated behind collect_records because it was the dominant RAM/CPU cost at
        # high arrival rates and the perf-critical proactive scripts never call
        # compute_batch_means().
        if self.collect_records:
            cpu_pct = (cpu_used / self.total_cpu_capacity * 100) if self.total_cpu_capacity > 0 else 0
            ram_pct = (ram_used / self.total_ram_capacity * 100) if self.total_ram_capacity > 0 else 0
            self.resource_history.append({
                'time': now,
                'cpu_usage': cpu_used,
                'ram_usage': ram_used,
                'power_usage': power_used,
                'cpu_usage_percent': cpu_pct,
                'ram_usage_percent': ram_pct,
                'waiting_requests': self.waiting_requests_count,
                'processing_requests': self.request_running,
            })

    def update_idle_ram_stats(self):
        """Accumulate time-weighted warm/idle RAM (RAM-seconds).

        Flush-before-mutate: callers invoke this immediately *before* any change
        to warm-pool membership or a pooled container's Idle state, so the past
        interval is attributed to the RAM held during it. Only containers in
        idle_containers whose state == "Idle" are counted (they hold warm_ram);
        cold-start/processing RAM is excluded.
        """
        now = self.env.now
        idle_ram = sum(c.ram_current for c in self.idle_containers if c.state == "Idle")
        self.total_idle_ram_area += (now - self.last_idle_ram_update) * idle_ram
        self.last_idle_ram_update = now

    def get_free_ram_time(self):
        """Total warm/idle RAM-seconds accumulated so far (wasted idle RAM)."""
        self.update_idle_ram_stats()
        return self.total_idle_ram_area

    def get_latency_percentiles(self, p_lo=95, p_hi=99):
        """Return (p95, p99) of the end-to-end latency of finished requests over
        the whole (post-warmup) run, or (0.0, 0.0) if none completed."""
        import numpy as np
        lats = self.latency_stats['all_latencies']
        if not lats:
            return 0.0, 0.0
        arr = np.asarray(lats, dtype=float)
        return float(np.percentile(arr, p_lo)), float(np.percentile(arr, p_hi))

    def _steady_state_duration(self):
        """Return the duration over which steady-state metrics have been accumulated."""
        return max(0.0, self.env.now - self.warmup_time)

    def get_mean_cpu_usage(self):
        self.update_resource_stats()
        d = self._steady_state_duration()
        return self.total_cpu_usage_area / d if d > 0 else 0.0

    def get_mean_ram_usage(self):
        self.update_resource_stats()
        d = self._steady_state_duration()
        return self.total_ram_usage_area / d if d > 0 else 0.0

    def get_mean_power_usage(self):
        self.update_resource_stats()
        d = self._steady_state_duration()
        return self.total_energy_usage / d if d > 0 else 0.0

    def get_mean_processing_count(self):
        self.update_processing_stats()
        d = self._steady_state_duration()
        return self.total_processing_area / d if d > 0 else 0.0

    def get_mean_requests_in_system(self):
        self.update_waiting_stats()
        self.update_processing_stats()
        d = self._steady_state_duration()
        return ((self.total_waiting_area + self.total_processing_area) / d
                if d > 0 else 0.0)

    def record_request_info(self, request):
        """Append a summary record for a completed or rejected request."""
        self.request_records.append({
            'id': request.id,
            'arrival_time': request.arrival_time,
            'end_time': self.env.now,
            'status': request.state,
            'waiting_time': request.wait_time,
            'processing_time': max(0, request.end_service_time - request.start_service_time),
            'spawn_time': request.spawn_time,
        })

    # ------------------------------------------------------------------
    # Batch-means: per-batch metric values for steady-state CI estimation
    # ------------------------------------------------------------------

    def compute_batch_means(self, num_batches=20, warmup_frac=None):
        """Split [warmup, sim_time] into equal-time batches and compute per-batch metrics.

        By default the warmup boundary equals ``self.warmup_time`` (absolute).
        If *warmup_frac* is provided it overrides as a fraction of sim_end.

        Returns a dict mapping each metric name to a list of per-batch values.
        Batches with insufficient data (e.g. zero requests) are omitted from
        that metric's list. Treating batch means as approximately IID enables
        Student-t CIs on the steady-state mean.
        """
        import numpy as np

        sim_end = self.env.now
        warmup_end = sim_end * warmup_frac if warmup_frac is not None else self.warmup_time
        usable = sim_end - warmup_end
        if usable <= 0 or num_batches < 2:
            return {k: [] for k in
                    ['blocking_probability', 'latency', 'variance_latency',
                     'p99_latency', 'cpu_usage', 'ram_usage', 'power_usage']}
        batch_dur = usable / num_batches
        edges = [warmup_end + i * batch_dur for i in range(num_batches + 1)]

        # Per-batch buckets for request-level metrics
        gen_count = [0] * num_batches
        blocked_count = [0] * num_batches
        latencies_per_batch = [[] for _ in range(num_batches)]

        for r in self.request_records:
            arr = r['arrival_time']
            if arr < warmup_end or arr >= sim_end:
                continue
            b = min(int((arr - warmup_end) / batch_dur), num_batches - 1)
            gen_count[b] += 1
            if r['status'] == 'Rejected':
                blocked_count[b] += 1
            elif r['status'] == 'Finished':
                latencies_per_batch[b].append(r['waiting_time'] + r['processing_time'])

        # Time-weighted resource averages per batch from event-driven snapshots.
        # Each snapshot records resource levels at the moment they changed;
        # the value persists until the next snapshot.  We compute the
        # time-weighted average within each batch by accumulating
        # (value * duration) for every interval that overlaps the batch.
        cpu_area = [0.0] * num_batches
        ram_area = [0.0] * num_batches
        pow_area = [0.0] * num_batches
        batch_covered = [0.0] * num_batches  # total time covered by snapshots per batch

        # Filter snapshots to the usable window and pair each with the next
        usable_snaps = [s for s in self.resource_history if s['time'] < sim_end]
        for i, snap in enumerate(usable_snaps):
            t_start = snap['time']
            t_end = usable_snaps[i + 1]['time'] if i + 1 < len(usable_snaps) else sim_end
            if t_end <= warmup_end or t_start >= sim_end:
                continue
            # Clamp to [warmup_end, sim_end]
            t_start = max(t_start, warmup_end)
            t_end = min(t_end, sim_end)
            # Determine which batches this interval overlaps
            b_start = max(0, int((t_start - warmup_end) / batch_dur))
            b_end = min(num_batches - 1, int((t_end - warmup_end - 1e-12) / batch_dur))
            for b in range(b_start, b_end + 1):
                batch_lo = warmup_end + b * batch_dur
                batch_hi = warmup_end + (b + 1) * batch_dur
                overlap_lo = max(t_start, batch_lo)
                overlap_hi = min(t_end, batch_hi)
                dt = overlap_hi - overlap_lo
                if dt > 0:
                    cpu_area[b] += snap['cpu_usage'] * dt
                    ram_area[b] += snap['ram_usage'] * dt
                    pow_area[b] += snap.get('power_usage', 0.0) * dt
                    batch_covered[b] += dt

        result = {
            'blocking_probability': [blocked_count[b] / gen_count[b]
                                     for b in range(num_batches) if gen_count[b] > 0],
            'latency': [float(np.mean(latencies_per_batch[b]))
                        for b in range(num_batches) if len(latencies_per_batch[b]) > 0],
            'variance_latency': [float(np.var(latencies_per_batch[b], ddof=1))
                                 for b in range(num_batches) if len(latencies_per_batch[b]) > 1],
            'p99_latency': [float(np.percentile(latencies_per_batch[b], 99))
                            for b in range(num_batches) if len(latencies_per_batch[b]) >= 5],
            'cpu_usage': [cpu_area[b] / batch_covered[b]
                          for b in range(num_batches) if batch_covered[b] > 0],
            'ram_usage': [ram_area[b] / batch_covered[b]
                          for b in range(num_batches) if batch_covered[b] > 0],
            'power_usage': [pow_area[b] / batch_covered[b]
                            for b in range(num_batches) if batch_covered[b] > 0],
        }
        return result
