import simpy
import random
from variables import APPLICATIONS, VERBOSE, CLUSTER_STRATEGY   
from Container import Container

class Server:
    """Represents a physical server with resource capacities."""
    def __init__(self, env, cluster, server_id, config):
        self.env = env
        self.id = server_id
        self.cluster = cluster
        # self.verbose = verbose  # Get verbose flag from cluster
        # Normal variables to represent available resources
        if CLUSTER_STRATEGY == "centralized_cloud":
            self.cpu_capacity = 1000000000  # Very large CPU capacity
            self.ram_capacity = 1000000000
            
        else:
            self.cpu_capacity = config['server_cpu']  # Store the maximum capacity
            self.ram_capacity = config['server_ram']  # Store the maximum capacity
        self.power_max = config['power_max']  # Max power consumption
        self.power_min = config['power_min']  # Min power consumption
        self.cpu_reserve = self.cpu_capacity
        self.ram_reserve = self.ram_capacity
        self.cpu_real = self.cpu_capacity
        self.ram_real = self.ram_capacity
        self.spawn_time_factor = config['spawn_time_factor']  # Spawn time multiplier
        self.containers = [] # Keep track of containers running on this server
        # Add a lock to prevent race conditions during resource allocation
        # self.resource_lock = simpy.Resource(env, capacity=1)

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

    def spawn_container_process(self, system, request, cluster_name):

        """Simulates the time and resource acquisition for spawning a container."""
        if VERBOSE:
            print(f"{self.env.now:.2f} - Attempting to acquire resources on {self} for {request}...")

        try:
            # Request the lock to prevent race conditions
            # lock_request = self.resource_lock.request()
            # yield lock_request
            
            # Check available resources first before attempting to allocate
            if self.cpu_reserve < request.cpu_demand or self.ram_reserve < request.ram_demand or \
               self.cpu_real < request.cpu_warm or self.ram_real < request.ram_warm:
                print(f"{self.env.now:.2f} - Resource check failed just before acquiring on {self} for {request}.")
                # self.resource_lock.release(lock_request)
                self.env.exit(None)  # Signal spawn failure

            # Log levels before acquiring resources
            if VERBOSE:
                print(f"{self.env.now:.2f} - Server before spawn: {self}")
                print(f"{self.env.now:.2f} - Demanding resources: CPU_Real {request.cpu_warm:.1f}, RAM_Real {request.ram_warm:.1f}, CPU_Reserve {request.cpu_demand:.1f}, RAM_Reserve {request.ram_demand:.1f}")

            # Allocate resources directly from server variables
            self.cpu_real -= request.cpu_warm
            self.ram_real -= request.ram_warm
            self.cpu_reserve -= request.cpu_demand
            self.ram_reserve -= request.ram_demand

            # Log levels after acquiring resources
            if VERBOSE:
                print(f"{self.env.now:.2f} - Server after spawn: {self}")
            
            # Release the lock after allocation
            # self.resource_lock.release(lock_request)

            # Determine spawn time based on app type and cluster type
            base_spawn_time = APPLICATIONS[request.app_id]["base_spawn_time"]
            
            # Apply the cluster-specific spawn time factor
            spawn_time_mean = base_spawn_time * self.spawn_time_factor
            
            # Resources acquired, now wait for spawn time, which is exponential distributed
            spawn_time_real = random.expovariate(1.0/spawn_time_mean)
            if VERBOSE:
                print(f"{self.env.now:.2f} - Spawning container on {cluster_name} cluster with factor {self.spawn_time_factor}. Base time: {base_spawn_time:.2f}, Adjusted mean: {spawn_time_mean:.2f}")
                print(f"{self.env.now:.2f} - Waiting {spawn_time_real:.2f} time units for container to spawn...")
            yield self.env.timeout(spawn_time_real)

            # Record the spawn time for latency calculation
            request.spawn_time = spawn_time_real
            request.assigned_cluster = cluster_name
            # Spawning complete, create the container object with app_id and cluster
            container = Container(self.env, system, self.cluster, self, request)
            # container.state = "Running"  # Explicitly mark as reserved
            self.containers.append(container) # Track container on server
            if VERBOSE:
                print(f"{self.env.now:.2f} - Spawn Complete: Created {container}")
            
            # Add the newly spawned container to the idle pool immediately
            # system.add_idle_container(container, cluster_name)
            
            # Start its idle lifecycle - this will be interrupted when a request uses the container
            # container.idle_timeout_process = self.env.process(container.idle_lifecycle())
            
            return container  # Return the created container object

        except Exception as e:  # Catch other potential issues
            print(f"ERROR: {self.env.now:.2f} - Unexpected error during spawn for {request} on {self}: {e}")  # Always print errors
            # Make sure we release the lock if we have it and there's an error
            # if 'lock_request' in locals():
            #     try:
            #         self.resource_lock.release(lock_request)
            #     except:
            #         pass
            exit(1)
    
    def is_ON(self):
        """Checks if the server is currently powered on."""
        return self.cpu_real < self.cpu_capacity - 1 and self.ram_real < self.ram_capacity - 1
    
    def get_power(self):
        """Calculates the current power consumption based on resource usage."""
        if self.is_ON():
            return self.power_min + (self.power_max - self.power_min) * (1 - (self.cpu_real / self.cpu_capacity))
        else:
            return 0