import itertools
import simpy 
import random
# from variables import CONTAINER_ASSIGN_RATE

class Container:
    """Represents a container instance spawned on a server."""
    # Using itertools for unique container IDs across the system
    id_counter = itertools.count()

    def __init__(self, env, system, server, resource_info, is_cold_start=False):
        self.env = env
        self.state = "Idle" # Initial state
        self.id = next(Container.id_counter)
        self.system = system # Reference back to the main system
        self.server = server
        self.resource_info = resource_info
        self.is_cold_start = is_cold_start
        # Set initial resources based on whether this is a cold start
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
        self.idle_timeout_process = None # SimPy process for idle timeout
        self.assignment_process = None # Track the current assignment process

    def __str__(self):
        state = f"Serving {self.current_request.id}" if self.current_request else f"Idle since {self.idle_since:.2f}"
        return f"Cont_{self.id}(on Srv_{self.server.id}, State: {state})"
  
    def assign_request(self, request):

        delta_cpu = request.cpu_demand - self.cpu_current
        delta_ram = request.ram_demand - self.ram_current

        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} request asks for more resources (+CPU:{delta_cpu:.1f}, +RAM:{delta_ram:.1f})")
        
        self.system.update_resource_stats()
        
        # Attempt to allocate the new resources
        if not self.server.allocate_resources(delta_cpu, delta_ram):
            raise RuntimeError(f"{self.env.now:.2f} - Insufficient resources on {self.server} for {request}")
        
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} allocated resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")

        # Update the container's allocated resources
        self.cpu_current = request.cpu_demand
        self.ram_current = request.ram_demand
        
        self.current_request = request
        
        # Update the container's state
        self.state = "Active"
        # Update the request's service start time
        request.start_service_time = self.env.now
        
        return True
        
    def release_request(self):
        """Releases the finished request and marks the container as idle."""
        if not self.current_request:
            return # Nothing to release

        # Update resource statistics before changing resource allocation
        self.system.update_resource_stats()

        # Modify resources in the container
        delta_cpu = self.cpu_current - self.resource_info['warm_cpu']
        delta_ram = self.ram_current - self.resource_info['warm_ram']

        if(delta_cpu < 0 or delta_ram < 0):
            raise RuntimeError(f"{self.env.now:.2f} - {self}, release_request() trying to release more resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})")

        # Release resources back to the server
        try:
            self.server.cpu_real += delta_cpu
            self.server.ram_real += delta_ram
            if self.server.cpu_real > self.server.cpu_capacity or self.server.ram_real > self.server.ram_capacity:
                raise RuntimeError(f"{self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
        except Exception as e:
            raise RuntimeError(f"Releasing resources for {self}: {e}")

        # Update the container's allocated resources
        self.cpu_current = self.resource_info['warm_cpu']
        self.ram_current = self.resource_info['warm_ram']

        # Update the container's state
        self.state = "Idle"

        # Clear the current request
        finished_request = self.current_request
        finished_request.end_service_time = self.env.now
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {finished_request} finished service in {self}. Duration: {finished_request.end_service_time - finished_request.start_service_time:.2f}")
            print(f"{self.env.now:.2f} - {finished_request} releasing request")
        self.current_request = None
        self.idle_since = self.env.now
    
        # Start the idle timeout process
        if self not in self.system.idle_containers:
            self.release_resources()
        
       
    def release_resources(self):
        """Returns allocated CPU and RAM resources back to the server."""
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} idle timeout expired, removing container")
            
        # Update resource statistics before changing resource allocation
        self.system.update_resource_stats()
            
        # Use try-except blocks for robustness
        self.state = "Dead"
        self.server.cpu_real += self.cpu_current
        self.server.cpu_reserve += self.cpu_reserve
        if self.server.cpu_real > self.server.cpu_capacity + 0.1 or self.server.cpu_reserve > self.server.cpu_capacity + 0.1:
            raise RuntimeError(f"{self.env.now:.2f} - {self} released more CPU than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f})")
        
        self.server.ram_real += self.ram_current
        self.server.ram_reserve += self.ram_reserve
        if self.server.ram_real > self.server.ram_capacity + 0.1 or self.server.ram_reserve > self.server.ram_capacity + 0.1:
            raise RuntimeError(f"{self.env.now:.2f} - {self} released more RAM than server capacity (RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
       
        if self in self.server.containers:
            self.server.containers.remove(self)