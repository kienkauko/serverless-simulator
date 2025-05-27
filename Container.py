import itertools
import simpy 
import random
# from variables import CONTAINER_ASSIGN_RATE

class Container:
    """Represents a container instance spawned on a server."""
    # Using itertools for unique container IDs across the system
    id_counter = itertools.count()

    def __init__(self, env, system, server, resource_info):
        self.env = env
        self.state = "Idle_cpu" # Initial state
        self.id = next(Container.id_counter)
        self.system = system # Reference back to the main system
        self.server = server
        self.resource_info = resource_info
        self.cpu_current = resource_info['warm_cpu']
        self.ram_current = resource_info['warm_ram']
        self.cpu_reserve = resource_info['cpu_demand']
        self.ram_reserve = resource_info['ram_demand']
        self.current_request = None
        self.idle_since = -1
        self.idle_timeout_process = None # SimPy process for idle timeout
        self.assignment_process = None # Track the current assignment process

    def __str__(self):
        return f"Cont_{self.id}(on Srv_{self.server.id}, State: {self.state})"

    def assign_model(self, request):
        if self.current_request:
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} trying to assign a request to a model ")
            exit(1) # Should not happen with correct logic

        try:         
            if self.state != "Idle_cpu":
                print(f"{self.env.now:.2f} - ERROR: {self} is dead, cannot assign request {request}")
                exit(1) # Should not happen with correct logic
            # Modify resources in the container
            delta_cpu = request.cpu_warm_model - self.cpu_current
            delta_ram = request.ram_warm_model - self.ram_current
            if self.system.verbose:
                print(f"{self.env.now:.2f} - {self} request asks for model loading (+CPU:{delta_cpu:.1f}, +RAM:{delta_ram:.1f})")
            
            # lock_request = self.server.resource_lock.request()
            # yield lock_request
            try:
                # Update resource statistics before changing resource allocation
                self.system.update_resource_stats()               
                # Attempt to allocate the new resources
                if not self.server.allocate_resources(delta_cpu, delta_ram):
                    print(f"{self.env.now:.2f} - FATAL ERROR: Insufficient resources on {self.server} for {request}")
                    # Release the lock before returning
                    # self.server.resource_lock.release(lock_request)
                    exit(1)
                
                if self.system.verbose:
                    print(f"{self.env.now:.2f} - {self} allocated model (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")
                
                # Release the lock after successful allocation
                # self.server.resource_lock.release(lock_request)

                # Update the container's allocated resources
                self.cpu_current = request.cpu_warm_model
                self.ram_current = request.ram_warm_model
                
                # If this container was idle, cancel its removal timeout
                if self.idle_timeout_process and not self.idle_timeout_process.triggered:
                    if self.system.verbose:
                        print(f"{self.env.now:.2f} - Cancelling idle timeout for reused {self}")
                    self.idle_timeout_process.interrupt()
                    self.idle_timeout_process = None # Clear the process handle
                self.idle_since = -1 # No longer idle

                # Start to duration of the assignment process
                if self.system.distribution == 'exponential':
                    assign_time = random.expovariate(1.0 / self.system.load_model_time_mean)
                else:
                    assign_time = self.system.load_model_time_mean
                assign_time = random.expovariate(1.0 / assign_time)
                if self.system.verbose:
                    print(f"{self.env.now:.2f} - {self} assignment model started for {request}, expected time: {assign_time:.2f}")    
                # Simulate the assignment process
                self.state = "CPU_To_Model"
                yield self.env.timeout(assign_time)
                
                # Update the container's state
                # NOTE: we won't set to Idle_model here because the container
                # is still in the idle_container pool, set it to Idle_cpu will
                # accidentally let other requests to use it
                # self.state = "Idle_model"

                return True
            
            except Exception as e:
                # Make sure we release the lock even if there's an error
                print(f"{self.env.now:.2f} - ERROR in assign_request: {e}")
                exit(1)
                
        except simpy.Interrupt:
            # This will be triggered if the idle_timeout_process interrupts this process
            if self.system.verbose:
                print(f"{self.env.now:.2f} - {self} assignment process for {request} was interrupted by timeout")
            exit(1)
            return False
        
    def assign_request(self, request):
        """Assigns a request to this container. Returns a generator for SimPy process."""
        # print(f"{self.env.now:.2f} - Inside assign_request for {self} and {request}")
        
        if self.current_request:
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} trying to assign {request} while already serving {self.current_request}")
            exit(1) # Should not happen with correct logic

        try:         
            if self.state == "Dead":
                print(f"{self.env.now:.2f} - ERROR: {self} is dead, cannot assign request {request}")
                exit(1) # Should not happen with correct logic
            # Modify resources in the container
            if self.state == "Idle_cpu":
                yield self.env.process(self.assign_model(request))
    
            delta_cpu = request.cpu_demand - self.cpu_current
            delta_ram = request.ram_demand - self.ram_current
            if self.system.verbose:
                print(f"{self.env.now:.2f} - {self} request asks for more resources (+CPU:{delta_cpu:.1f}, +RAM:{delta_ram:.1f})")
            
            # lock_request = self.server.resource_lock.request()
            # yield lock_request           
            try:
                # Update resource statistics before changing resource allocation
                self.system.update_resource_stats()
                
                # Attempt to allocate the new resources
                if not self.server.allocate_resources(delta_cpu, delta_ram):
                    print(f"{self.env.now:.2f} - FATAL ERROR: Insufficient resources on {self.server} for {request}")
                    # Release the lock before returning
                    # self.server.resource_lock.release(lock_request)
                    exit(1)
                
                if self.system.verbose:
                    print(f"{self.env.now:.2f} - {self} allocated resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")

                # Release the lock after successful allocation
                # self.server.resource_lock.release(lock_request)

                # Update the container's allocated resources
                self.cpu_current = request.cpu_demand
                self.ram_current = request.ram_demand
                
                # If this container was idle, cancel its removal timeout
                if self.idle_timeout_process and not self.idle_timeout_process.triggered:
                    if self.system.verbose:
                        print(f"{self.env.now:.2f} - Cancelling idle timeout for reused {self}")
                    self.idle_timeout_process.interrupt()
                    self.idle_timeout_process = None # Clear the process handle
                self.idle_since = -1 # No longer idle
                
                # Start to duration of the assignment process
                assign_time_mean = self.system.load_request_time_mean
                if self.system.distribution == 'exponential':
                    assign_time = random.expovariate(1.0 / assign_time_mean)
                else:
                    assign_time = assign_time_mean            
                if self.system.verbose:
                    print(f"{self.env.now:.2f} - {self} assignment process started for {request}, expected time: {assign_time:.2f}")    
                
                # Simulate the assignment process
                self.state = "Model_To_Active"
                yield self.env.timeout(assign_time)

                # This container is free, no request is assigned, so we assign request here
                self.current_request = request
                
                # Update the container's state
                self.state = "Active"
                
                # Update the request's service start time
                request.start_service_time = self.env.now
                
                return True
            
            except Exception as e:
                # Make sure we release the lock even if there's an error
                print(f"{self.env.now:.2f} - ERROR in assign_request: {e}")
                # self.server.resource_lock.release(lock_request)
                exit(1)
                
        except simpy.Interrupt:
            # This will be triggered if the idle_timeout_process interrupts this process
            if self.system.verbose:
                print(f"{self.env.now:.2f} - {self} assignment process for {request} was interrupted by timeout")
            exit(1)

    def release_request(self):
        """Releases the finished request and marks the container as idle."""
        if not self.current_request:
            return # Nothing to release

        # Update resource statistics before changing resource allocation
        self.system.update_resource_stats()

        # Modify resources in the container
        delta_cpu = self.cpu_current - self.resource_info['warm_cpu_model']
        delta_ram = self.ram_current - self.resource_info['warm_ram_model']

        if(delta_cpu < 0 or delta_ram < 0):
            print(f"FATAL ERROR: {self.env.now:.2f} - {self}, release_request() trying to release more resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})")
            exit(1)

        # Release resources back to the server
        try:
            self.server.cpu_real += delta_cpu
            self.server.ram_real += delta_ram
            if self.server.cpu_real > self.server.cpu_capacity or self.server.ram_real > self.server.ram_capacity:
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
                exit(1)
        except Exception as e:
            print(f"FATAL ERROR: releasing resources for {self}: {e}")
            exit(1)

        # Update the container's allocated resources
        self.cpu_current = self.resource_info['warm_cpu_model']
        self.ram_current = self.resource_info['warm_ram_model']

        # Update the container's state
        self.state = "Idle_model"

        # Clear the current request
        finished_request = self.current_request
        finished_request.end_service_time = self.env.now
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {finished_request} finished service in {self}. Duration: {finished_request.end_service_time - finished_request.start_service_time:.2f}")
            print(f"{self.env.now:.2f} - {finished_request} releasing request")
        self.current_request = None
        self.idle_since = self.env.now

        # Put the container into the idle pool
        self.system.add_idle_container(self)

        # Start the idle timeout process
        self.idle_timeout_process = self.env.process(self.system.container_idle_lifecycle(self))

    def release_model(self):
        """Releases the finished request and marks the container as idle."""
        if self.system.verbose:
            print(f"{self.env.now:.2f} - releasing model")
        # Update resource statistics before changing resource allocation
        self.system.update_resource_stats()

        # Modify resources in the container
        delta_cpu = self.cpu_current - self.resource_info['warm_cpu']
        delta_ram = self.ram_current - self.resource_info['warm_ram']

        if(delta_cpu < 0 or delta_ram < 0):
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} release_model() trying to release more resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})")
            exit(1)

        # Release resources back to the server
        try:
            self.server.cpu_real += delta_cpu
            self.server.ram_real += delta_ram
            if self.server.cpu_real > self.server.cpu_capacity or self.server.ram_real > self.server.ram_capacity:
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
                exit(1)
        except Exception as e:
            print(f"FATAL ERROR: releasing resources for {self}: {e}")
            exit(1)

        # Update the container's allocated resources
        self.cpu_current = self.resource_info['warm_cpu']
        self.ram_current = self.resource_info['warm_ram']

        # Update the container's state
        self.state = "Idle_cpu"

        # Put the container into the idle pool
        self.system.add_idle_container(self)
        
       
       

    def release_resources(self):
        """Returns allocated CPU and RAM resources back to the server."""
        if self.system.verbose:
            print(f"{self.env.now:.2f} - {self} idle timeout expired, removing container")
            
        # Update resource statistics before changing resource allocation
        self.system.update_resource_stats()
            
        # Use try-except blocks for robustness
        self.state = "Dead"
        try:
            self.server.cpu_real += self.cpu_current
            self.server.cpu_reserve += self.cpu_reserve
            if self.server.cpu_real > self.server.cpu_capacity or self.server.cpu_reserve > self.server.cpu_capacity:
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more CPU than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f})")
                exit(1)
        except Exception as e:
             print(f"ERROR releasing CPU for {self}: {e}")
        try:
            self.server.ram_real += self.ram_current
            self.server.ram_reserve += self.ram_reserve
            if self.server.ram_real > self.server.ram_capacity or self.server.ram_reserve > self.server.ram_capacity:
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more RAM than server capacity (RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
                exit(1)
        except Exception as e:
             print(f"ERROR releasing RAM for {self}: {e}")
        # Remove from server's list
        if self in self.server.containers:
            self.server.containers.remove(self)