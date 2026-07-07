import itertools

import simpy


class Container:
    """Container with a per-instance idle timeout.

    After serving a request the container becomes idle and a timer starts.
    If reused within ``idle_timeout`` seconds the timer is cancelled; otherwise
    the container is evicted and its resources released.
    """

    id_counter = itertools.count()

    def __init__(self, env, system, server, resource_info, idle_timeout, is_cold_start=True):
        self.env = env
        self.state = "Idle"
        self.id = next(Container.id_counter)
        self.system = system
        self.server = server
        self.resource_info = resource_info
        self.is_cold_start = is_cold_start
        if self.is_cold_start:
            self.cpu_current = resource_info['cold_start_cpu']
            self.ram_current = resource_info['cold_start_ram']
        else:
            self.cpu_current = resource_info['warm_cpu']
            self.ram_current = resource_info['warm_ram']
        self.cpu_reserve = resource_info['cpu_demand']
        self.ram_reserve = resource_info['ram_demand']
        self.current_request = None
        self.idle_since = -1
        self.idle_timeout = idle_timeout
        self.idle_timer_proc = None

    def __str__(self):
        state = (f"Serving {self.current_request.id}"
                 if self.current_request
                 else f"Idle since {self.idle_since:.2f}")
        return f"Cont_{self.id}(on Srv_{self.server.id}, State: {state})"

    def assign_request(self, request):
        """Allocate the additional resources needed by *request* and mark container active."""
        delta_cpu = request.cpu_demand - self.cpu_current
        delta_ram = request.ram_demand - self.ram_current

        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} request asks for more resources "
                  f"(+CPU:{delta_cpu:.1f}, +RAM:{delta_ram:.1f})")

        self.system.update_resource_stats()

        if not self.server.allocate_resources(delta_cpu, delta_ram):
            raise RuntimeError(
                f"{self.env.now:.2f} - Insufficient resources on {self.server} for {request}"
            )

        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} allocated resources "
                  f"(CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")

        self.cpu_current = request.cpu_demand
        self.ram_current = request.ram_demand
        self.current_request = request
        self.state = "Active"
        request.start_service_time = self.env.now
        return True

    def release_request(self):
        """Release the finished request, return to idle, and start the idle timer."""
        if not self.current_request:
            return

        self.system.update_resource_stats()

        delta_cpu = self.cpu_current - self.resource_info['warm_cpu']
        delta_ram = self.ram_current - self.resource_info['warm_ram']

        if delta_cpu < 0 or delta_ram < 0:
            raise RuntimeError(
                f"{self.env.now:.2f} - {self} release_request() trying to release more "
                f"resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})"
            )

        try:
            self.server.cpu_real += delta_cpu
            self.server.ram_real += delta_ram
            if (self.server.cpu_real > self.server.cpu_capacity or
                    self.server.ram_real > self.server.ram_capacity):
                raise RuntimeError(
                    f"{self.env.now:.2f} - {self} released more resources than server capacity "
                    f"(CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, "
                    f"RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})"
                )
        except Exception as e:
            raise RuntimeError(f"Releasing resources for {self}: {e}")

        self.cpu_current = self.resource_info['warm_cpu']
        self.ram_current = self.resource_info['warm_ram']
        self.state = "Idle"

        finished_request = self.current_request
        finished_request.end_service_time = self.env.now
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {finished_request} finished service in {self}. "
                  f"Duration: {finished_request.end_service_time - finished_request.start_service_time:.2f}")
        self.current_request = None
        self.idle_since = self.env.now

        # Dynamic-pool eviction policy: every container enters the idle pool
        # with its own timer. If idle_timeout <= 0, evict immediately.
        if self.idle_timeout <= 0:
            self.release_resources()
        else:
            self.system.update_idle_stats()
            self.system.idle_containers.append(self)
            self.idle_timer_proc = self.env.process(self._idle_timeout_process())

    def cancel_idle_timer(self):
        """Cancel the pending idle-eviction timer (called on reuse)."""
        if self.idle_timer_proc is not None and self.idle_timer_proc.is_alive:
            self.idle_timer_proc.interrupt()
        self.idle_timer_proc = None

    def _idle_timeout_process(self):
        """Wait for the idle-eviction delay (fixed or memoryless, per the
        system's idle-distribution); if still idle, evict the container."""
        try:
            yield self.env.timeout(self.system.sample_idle_timeout(self.idle_timeout))
            if self.state != "Idle":
                return
            if self in self.system.idle_containers:
                self.system.update_idle_stats()
                self.system.idle_containers.remove(self)
            self.idle_timer_proc = None
            self.release_resources()
        except simpy.Interrupt:
            return

    def release_resources(self):
        """Deallocate all server resources held by this container (eviction)."""
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} evicted, releasing resources")

        self.system.update_resource_stats()
        self.state = "Dead"
        self.server.cpu_real += self.cpu_current
        self.server.cpu_reserve += self.cpu_reserve
        if (self.server.cpu_real > self.server.cpu_capacity + 0.1 or
                self.server.cpu_reserve > self.server.cpu_capacity + 0.1):
            raise RuntimeError(
                f"{self.env.now:.2f} - {self} released more CPU than server capacity "
                f"(CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f})"
            )

        self.server.ram_real += self.ram_current
        self.server.ram_reserve += self.ram_reserve
        if (self.server.ram_real > self.server.ram_capacity + 0.1 or
                self.server.ram_reserve > self.server.ram_capacity + 0.1):
            raise RuntimeError(
                f"{self.env.now:.2f} - {self} released more RAM than server capacity "
                f"(RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})"
            )

        if self in self.server.containers:
            self.server.containers.remove(self)
