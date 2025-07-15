import simpy
class Server:
    """Represents a physical server with resource capacities."""
    def __init__(self, env, server_id, config):
        self.env = env
        self.id = server_id
        # Normal variables to represent available resources
        self.cpu_reserve = config["cpu_capacity"]
        self.ram_reserve = config["ram_capacity"]
        self.cpu_real = config["cpu_capacity"]
        self.ram_real = config["ram_capacity"]
        self.cpu_capacity = config["cpu_capacity"]  # Store the maximum capacity
        self.ram_capacity = config["ram_capacity"]  # Store the maximum capacity
        self.containers = [] # Keep track of containers running on this server
        self.power_max = config["power_max"]  # Maximum power consumption
        self.power_min = self.power_max*config["power_min_scale"]  # Minimum power consumption
        # Add a lock to prevent race conditions during resource allocation
        self.resource_lock = simpy.Resource(env, capacity=1)

    def __str__(self):
        return (f"Server_{self.id}("
                f"CPU_Reserve:{self.cpu_reserve:.1f}/{self.cpu_capacity:.1f}, "
                f"RAM_Reserve:{self.ram_reserve:.1f}/{self.ram_capacity:.1f}, "
                f"CPU_Real:{self.cpu_real:.1f}/{self.cpu_capacity:.1f}, "
                f"RAM_Real:{self.ram_real:.1f}/{self.ram_capacity:.1f})")

    def has_capacity(self, resource_info):
        """Checks if the server currently has enough free resources."""       
        return self.cpu_reserve >= resource_info['cpu_demand'] and self.ram_reserve >= resource_info['ram_demand'] and self.cpu_real >= resource_info['warm_cpu_model'] and self.ram_real >= resource_info['warm_ram_model']
    
    def allocate_resources(self, delta_cpu, delta_ram):
        """Safely allocates CPU and RAM resources if available.
        Returns True if allocation was successful, False otherwise."""
        if self.cpu_real >= delta_cpu and self.ram_real >= delta_ram:
            self.cpu_real -= delta_cpu
            self.ram_real -= delta_ram
            return True
        return False
    
    def is_ON(self):
        """Checks if the server is currently powered on."""
        return self.cpu_real < self.cpu_capacity - 1 and self.ram_real < self.ram_capacity - 1
    
    def power_consumption(self):
        """Calculates the current power consumption based on resource usage."""
        if self.is_ON():
            return self.power_min + (self.power_max - self.power_min) * (1 - (self.cpu_real / self.cpu_capacity))
        else:
            return 0