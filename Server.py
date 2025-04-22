import simpy
class Server:
    """Represents a physical server with resource capacities."""
    def __init__(self, env, server_id, cpu_capacity, ram_capacity):
        self.env = env
        self.id = server_id
        # Normal variables to represent available resources
        self.cpu_reserve = cpu_capacity
        self.ram_reserve = ram_capacity
        self.cpu_real = cpu_capacity
        self.ram_real = ram_capacity
        self.cpu_capacity = cpu_capacity  # Store the maximum capacity
        self.ram_capacity = ram_capacity  # Store the maximum capacity
        self.containers = [] # Keep track of containers running on this server
        # Add a lock to prevent race conditions during resource allocation
        self.resource_lock = simpy.Resource(env, capacity=1)

    def __str__(self):
        return (f"Server_{self.id}("
                f"CPU_Reserve:{self.cpu_reserve:.1f}/{self.cpu_capacity:.1f}, "
                f"RAM_Reserve:{self.ram_reserve:.1f}/{self.ram_capacity:.1f}, "
                f"CPU_Real:{self.cpu_real:.1f}/{self.cpu_capacity:.1f}, "
                f"RAM_Real:{self.ram_real:.1f}/{self.ram_capacity:.1f})")

    def has_capacity(self, cpu_demand, ram_demand, cpu_warm, ram_warm):
        """Checks if the server currently has enough free resources."""
        
        return self.cpu_reserve >= cpu_demand and self.ram_reserve >= ram_demand and self.cpu_real >= cpu_warm and self.ram_real >= ram_warm
    
    def allocate_resources(self, delta_cpu, delta_ram):
        """Safely allocates CPU and RAM resources if available.
        Returns True if allocation was successful, False otherwise."""
        if self.cpu_real >= delta_cpu and self.ram_real >= delta_ram:
            self.cpu_real -= delta_cpu
            self.ram_real -= delta_ram
            return True
        return False