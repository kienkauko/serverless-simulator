class Server:
    """Represents a physical server with fixed CPU and RAM capacities."""

    def __init__(self, env, server_id, config):
        self.env = env
        self.id = server_id
        self.cpu_capacity = config['cpu_capacity']
        self.ram_capacity = config['ram_capacity']
        self.cpu_reserve = config['cpu_capacity']   # Headroom reserved for future requests
        self.ram_reserve = config['ram_capacity']
        self.cpu_real = config['cpu_capacity']       # Currently available real resources
        self.ram_real = config['ram_capacity']
        self.peak_power = config['peak_power']
        self.power_scale = config['power_scale']
        self.containers = []

    def __str__(self):
        return (f"Server_{self.id}("
                f"CPU_Reserve:{self.cpu_reserve:.1f}/{self.cpu_capacity:.1f}, "
                f"RAM_Reserve:{self.ram_reserve:.1f}/{self.ram_capacity:.1f}, "
                f"CPU_Real:{self.cpu_real:.1f}/{self.cpu_capacity:.1f}, "
                f"RAM_Real:{self.ram_real:.1f}/{self.ram_capacity:.1f})")

    def has_capacity(self, resource_info):
        """Return True if the server can currently host a new container."""
        return (self.cpu_reserve >= resource_info['cpu_demand'] and
                self.ram_reserve >= resource_info['ram_demand'] and
                self.cpu_real >= resource_info['warm_cpu'] and
                self.ram_real >= resource_info['warm_ram'])

    def allocate_resources(self, delta_cpu, delta_ram):
        """Deduct delta_cpu/delta_ram from real resources. Returns False if insufficient."""
        if self.cpu_real >= delta_cpu and self.ram_real >= delta_ram:
            self.cpu_real -= delta_cpu
            self.ram_real -= delta_ram
            return True
        return False

    def is_on(self):
        """Return True if the server is currently handling any workload."""
        return self.cpu_real < self.cpu_capacity - 1 or self.ram_real < self.ram_capacity - 1

    def current_power(self):
        """Estimate instantaneous power draw based on CPU utilisation."""
        if self.is_on():
            cpu_util = (self.cpu_capacity - self.cpu_real) / self.cpu_capacity
            return self.peak_power * (self.power_scale + cpu_util * (1 - self.power_scale))
        return 0.0
