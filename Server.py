import simpy
import random
from variables import APPLICATIONS
from Container import Container

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

    def spawn_container_process(self, system, request, path):
        """Simulates the time and resource acquisition for spawning a container."""
        print(f"{self.env.now:.2f} - Attempting to acquire resources on {self} for {request}...")

        try:
            # Request the lock to prevent race conditions
            lock_request = self.resource_lock.request()
            yield lock_request
            
            # Check available resources first before attempting to allocate
            if self.cpu_reserve < request.cpu_demand or self.ram_reserve < request.ram_demand or \
               self.cpu_real < request.cpu_warm or self.ram_real < request.ram_warm:
                print(f"{self.env.now:.2f} - Resource check failed just before acquiring on {self} for {request}.")
                self.resource_lock.release(lock_request)
                self.env.exit(None)  # Signal spawn failure

            # Log levels before acquiring resources
            print(f"{self.env.now:.2f} - Server before spawn: {self}")
            print(f"{self.env.now:.2f} - Demanding resources: CPU_Real {request.cpu_warm:.1f}, RAM_Real {request.ram_warm:.1f}, CPU_Reserve {request.cpu_demand:.1f}, RAM_Reserve {request.ram_demand:.1f}")

            # Allocate resources directly from server variables
            self.cpu_real -= request.cpu_warm
            self.ram_real -= request.ram_warm
            self.cpu_reserve -= request.cpu_demand
            self.ram_reserve -= request.ram_demand

            # Log levels after acquiring resources
            print(f"{self.env.now:.2f} - Server after spawn: {self}")
            
            # Release the lock after allocation
            self.resource_lock.release(lock_request)

            # Determine spawn time based on app type
            spawn_time_mean = APPLICATIONS[request.app_id]["spawn_time"]
            
            # Resources acquired, now wait for spawn time, which is exponential distributed
            spawn_time_real = random.expovariate(1.0/spawn_time_mean)
            print(f"Waiting {spawn_time_real:.2f} time units...")
            yield self.env.timeout(spawn_time_real)

            # Record the spawn time for latency calculation
            request.spawn_time = spawn_time_real

            # Spawning complete, create the container object with app_id
            container = Container(self.env, system, self, request.cpu_warm, request.ram_warm, 
                                 request.cpu_demand, request.ram_demand, app_id=request.app_id)
            container.state = "Idle"  # Explicitly mark as idle
            self.containers.append(container) # Track container on server
            print(f"{self.env.now:.2f} - Spawn Complete: Created {container}")
            
            # Add the newly spawned container to the idle pool immediately
            system.add_idle_container(container)
            
            # Start its idle lifecycle - this will be interrupted when a request uses the container
            container.idle_timeout_process = self.env.process(container.idle_lifecycle())
            
            return container  # Return the created container object

        except Exception as e:  # Catch other potential issues
            print(f"ERROR: {self.env.now:.2f} - Unexpected error during spawn for {request} on {self}: {e}")
            # Make sure we release the lock if we have it and there's an error
            if 'lock_request' in locals():
                try:
                    self.resource_lock.release(lock_request)
                except:
                    pass
            exit(1)