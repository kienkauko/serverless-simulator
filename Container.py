import itertools
import simpy 
import random
from variables import CONTAINER_ASSIGN_RATE

class Container:
    """Represents a container instance spawned on a server."""
    # Using itertools for unique container IDs across the system
    id_counter = itertools.count()

    def __init__(self, env, system, server, cpu_alloc, ram_alloc, cpu_reserve, ram_reserve):
        self.env = env
        self.state = "Idle" # Initial state
        self.id = next(Container.id_counter)
        self.system = system # Reference back to the main system
        self.server = server
        self.cpu_alloc = cpu_alloc
        self.ram_alloc = ram_alloc
        self.cpu_reserve = cpu_reserve
        self.ram_reserve = ram_reserve
        self.current_request = None
        self.idle_since = -1
        self.idle_timeout_process = None # SimPy process for idle timeout
        self.assignment_process = None # Track the current assignment process

    def __str__(self):
        state = f"Serving {self.current_request.id}" if self.current_request else f"Idle since {self.idle_since:.2f}"
        return f"Cont_{self.id}(on Srv_{self.server.id}, State: {state})"

    def assign_request(self, request):
        """Assigns a request to this container. Returns a generator for SimPy process."""
        # print(f"{self.env.now:.2f} - Inside assign_request for {self} and {request}")
        
        if self.current_request:
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} trying to assign {request} while already serving {self.current_request}")
            exit(1) # Should not happen with correct logic

        try:
            # Add a small timeout to make this a generator function
            assign_time = random.expovariate(CONTAINER_ASSIGN_RATE)
            yield self.env.timeout(assign_time)
            
            if self.state == "Dead":
                print(f"{self.env.now:.2f} - ERROR: {self} is dead, cannot assign request {request}")
                return False # Should not happen with correct logic
            # Modify resources in the container
            delta_cpu = request.cpu_demand - self.cpu_alloc
            delta_ram = request.ram_demand - self.ram_alloc
            print(f"{self.env.now:.2f} - {self} request asks for more resources (+CPU:{delta_cpu:.1f}, +RAM:{delta_ram:.1f})")
            
            # Use the server's resource lock to prevent race conditions - request the lock
            lock_request = self.server.resource_lock.request()
            yield lock_request
            
            try:
                # Verify the server has enough resources before proceeding
                # print(f"{self.env.now:.2f} - current server resources: {self.server}")
                
                # Attempt to allocate the new resources
                if not self.server.allocate_resources(delta_cpu, delta_ram):
                    print(f"{self.env.now:.2f} - ERROR: Insufficient resources on {self.server} for {request}")
                    # Release the lock before returning
                    self.server.resource_lock.release(lock_request)
                    return False
                
                print(f"{self.env.now:.2f} - {self} allocated resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")
                
                # Release the lock after successful allocation
                self.server.resource_lock.release(lock_request)
                
                # This container is free, no request is assigned, so we assign request here
                self.current_request = request
                    
                # Update the container's allocated resources
                self.cpu_alloc = request.cpu_demand
                self.ram_alloc = request.ram_demand
                
                # Update the container's state
                self.state = "Active"
                
                # Update the request's service start time
                request.start_service_time = self.env.now
                self.idle_since = -1 # No longer idle
                
                # If this container was idle, cancel its removal timeout
                if self.idle_timeout_process and not self.idle_timeout_process.triggered:
                    print(f"{self.env.now:.2f} - Cancelling idle timeout for reused {self}")
                    self.idle_timeout_process.interrupt()
                    self.idle_timeout_process = None # Clear the process handle
                    
                return True
            
            except Exception as e:
                # Make sure we release the lock even if there's an error
                print(f"{self.env.now:.2f} - ERROR in assign_request: {e}")
                self.server.resource_lock.release(lock_request)
                return False
                
        except simpy.Interrupt:
            # This will be triggered if the idle_timeout_process interrupts this process
            print(f"{self.env.now:.2f} - {self} assignment process for {request} was interrupted by timeout")
            return False

    def release_request(self):
        """Releases the finished request and marks the container as idle."""
        if not self.current_request:
            return # Nothing to release

        # Modify resources in the container
        delta_cpu = self.cpu_alloc - self.current_request.cpu_warm
        delta_ram = self.ram_alloc - self.current_request.ram_warm

        if(delta_cpu < 0 or delta_ram < 0):
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} trying to release more resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})")
            exit(1)

        # Release resources back to the server
        try:
            self.server.cpu_real += delta_cpu
            self.server.ram_real += delta_ram
        except Exception as e:
            print(f"FATAL ERROR: releasing resources for {self}: {e}")
            exit(1)

        # Update the container's allocated resources
        self.cpu_alloc = self.current_request.cpu_warm
        self.ram_alloc = self.current_request.ram_warm

        # Update the container's state
        self.state = "Idle"

        # Clear the current request
        finished_request = self.current_request
        finished_request.end_service_time = self.env.now
        print(f"{self.env.now:.2f} - {finished_request} finished service in {self}. Duration: {finished_request.end_service_time - finished_request.start_service_time:.2f}")
        print(f"{self.env.now:.2f} - {finished_request} releasing request")
        self.current_request = None
        self.idle_since = self.env.now
        # Put the container into the idle pool
        self.system.add_idle_container(self)
        # Start the idle timeout process
        self.idle_timeout_process = self.env.process(self.system.container_idle_lifecycle(self))

    def release_resources(self):
        """Returns allocated CPU and RAM resources back to the server."""
        print(f"{self.env.now:.2f} - {self} idle timeout expired, removing container")
        # Use try-except blocks for robustness
        self.state = "Dead"
        try:
            self.server.cpu_real += self.cpu_alloc
            self.server.cpu_reserve += self.cpu_reserve
        except Exception as e:
             print(f"ERROR releasing CPU for {self}: {e}")
        try:
            self.server.ram_real += self.ram_alloc
            self.server.ram_reserve += self.ram_reserve
        except Exception as e:
             print(f"ERROR releasing RAM for {self}: {e}")
        # Remove from server's list
        if self in self.server.containers:
            self.server.containers.remove(self)