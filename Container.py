import itertools
import simpy 
import random
# from variables import CONTAINER_ASSIGN_RATE
import variables 

class Container:
    """Represents a container instance spawned on a server."""
    # Using itertools for unique container IDs across the system
    id_counter = itertools.count()

    def __init__(self, env, system, cluster, server, request):
        self.env = env
        self.state = "Assigned" # Initial state
        self.id = next(Container.id_counter)
        self.system = system # Reference back to the main system
        self.server = server
        self.cluster = cluster
        self.cpu_alloc = request.cpu_warm
        self.ram_alloc = request.ram_warm
        self.cpu_reserve = request.cpu_demand
        self.ram_reserve = request.ram_demand
        self.current_request = request  # Track the current request being served
        self.time_out = None  # Time after which the container should be removed if idle
        self.idle_since = -1
        self.idle_timeout_process = None # SimPy process for idle timeout
        self.app_id = request.app_id # Application ID this container belongs to
        # self.verbose = system.verbose  # Get verbose flag from system
        # self.cluster_name = cluster_name # Cluster this container belongs to

    def __str__(self):
        state = f"Serving {self.current_request.id}" if self.current_request else f"Idle since {self.idle_since:.2f}"
        app_info = f" [App: {self.app_id}]" if self.app_id else ""
        return f"Cont_{self.id}(on Srv_{self.server.id}, State: {state}){app_info}"

    def scale_for_request(self):
        """Scales container resources for the current request."""
        if not self.current_request:
            print(f"ERROR: {self.env.now:.2f} - scale_for_request called for {self} with no request!")
            return False
            
        request = self.current_request
            
        # Modify resources in the container
        delta_cpu = request.cpu_demand - self.cpu_alloc
        delta_ram = request.ram_demand - self.ram_alloc
        if variables.VERBOSE:
            print(f"{self.env.now:.2f} - {self} request asks for more resources (+CPU:{delta_cpu:.1f}, +RAM:{delta_ram:.1f})")
        
        # Use the server's resource lock to prevent race conditions - request the lock
        # lock_request = self.server.resource_lock.request()
        # yield lock_request
        # Update resource stats before allocation
        self.cluster.update_resource_stats()
        try:            
            # Attempt to allocate the new resources
            if not self.server.allocate_resources(delta_cpu, delta_ram):
                print(f"{self.env.now:.2f} - ERROR: Insufficient resources on {self.server} for {request}")
                # Release the lock before returning
                # self.server.resource_lock.release(lock_request)
                # Reset the container state and clear the current_request since scaling failed
                # self.state = "Idle"
                # self.current_request = None
                exit(1)
            
            if variables.VERBOSE:
                print(f"{self.env.now:.2f} - {self} allocated resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")
            
            # Release the lock after successful allocation
            # self.server.resource_lock.release(lock_request)
                
            # Update the container's allocated resources
            self.cpu_alloc = request.cpu_demand
            self.ram_alloc = request.ram_demand
            
            # Update the request's service start time
            request.start_service_time = self.env.now
                
            return True
        
        except Exception as e:
            # Make sure we release the lock even if there's an error
            print(f"{self.env.now:.2f} - ERROR in scale_for_request: {e}")
            # self.server.resource_lock.release(lock_request)
            # Reset the container state and clear the current_request since scaling failed
            self.state = "Idle"
            self.current_request = None
            return False

    def release_request(self):
        """Releases the finished request and marks the container as idle."""
        if not self.current_request:
            print("FATAL ERROR: Container Class: release_request called with no current request!")
            exit(1) # Nothing to release

        # Use the server's resource lock to prevent race conditions
        # lock_request = self.server.resource_lock.request()
        # yield lock_request
        
        try:
            # Modify resources in the container
            delta_cpu = self.cpu_alloc - self.current_request.cpu_warm
            delta_ram = self.ram_alloc - self.current_request.ram_warm

            if(delta_cpu <= 0 or delta_ram <= 0):
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} trying to release more resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})")
                exit(1)
                
            # Update resource stats before releasing
            self.cluster.update_resource_stats()
            # Release resources back to the server with bounds checking
            self.server.cpu_real += delta_cpu
            self.server.ram_real += delta_ram
            if(self.server.cpu_real > self.server.cpu_capacity or self.server.ram_real > self.server.ram_capacity):
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
                exit(1)
            if(self.server.cpu_reserve > self.server.cpu_capacity or self.server.ram_reserve > self.server.ram_capacity):
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
                exit(1)
        except Exception as e:
            print(f"FATAL ERROR: releasing resources for {self}: {e}")
            exit(1)

        # Update the container's allocated resources
        self.cpu_alloc = self.current_request.cpu_warm
        self.ram_alloc = self.current_request.ram_warm

        # Release the lock after resource update
        # self.server.resource_lock.release(lock_request)
        # Clear the current request
        # finished_request = self.current_request
        # finished_request.end_service_time = self.env.now
        # print(f"{self.env.now:.2f} - {finished_request} finished service in {self}. Duration: {finished_request.end_service_time - finished_request.start_service_time:.2f}")
        # print(f"{self.env.now:.2f} - {finished_request} releasing resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) from {self.server}")
        # Update the container's state
        self.state = "Idle"
        self.idle_since = self.env.now
        self.current_request = None
        # Start the idle timeout process
        self.idle_timeout_process = self.env.process(self.idle_lifecycle())

    def release_resources(self):
        """Returns allocated CPU and RAM resources back to the server."""
        if variables.VERBOSE:
            print(f"{self.env.now:.2f} - {self} releasing resources (CPU:{self.cpu_alloc:.1f}, RAM:{self.ram_alloc:.1f}) from {self.server}")
        
        # Use the server's resource lock to prevent race conditions
        # lock_request = self.server.resource_lock.request()
        # yield lock_request
        
        # Update resource stats before releasing
        self.cluster.update_resource_stats()
        # Release all resources with bounds checking
        self.server.cpu_real += self.cpu_alloc
        self.server.ram_real += self.ram_alloc
        self.server.cpu_reserve += self.cpu_reserve
        self.server.ram_reserve += self.ram_reserve
        
        if(self.server.cpu_real > self.server.cpu_capacity or self.server.ram_real > self.server.ram_capacity):
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
            exit(1)
        
        if(self.server.cpu_reserve > self.server.cpu_capacity or self.server.ram_reserve > self.server.ram_capacity):
            print(f"FATAL ERROR: {self.env.now:.2f} - {self} released more resources than server capacity (CPU:{self.server.cpu_real:.1f}/{self.server.cpu_capacity:.1f}, RAM:{self.server.ram_real:.1f}/{self.server.ram_capacity:.1f})")
            exit(1)

        # self.server.resource_lock.release(lock_request)


    def service_lifecycle(self):
        """Simulates the request processing time within a container."""
        # if not self.current_request:
        #     print(f"ERROR: {self.env.now:.2f} - service_lifecycle called for {self} with no request!")
        #     return
        
        # Calculate and record the waiting time
        if self.current_request.waiting_start_time != -1:
            self.current_request.waiting_time = self.env.now - self.current_request.waiting_start_time
            # print(f"{self.env.now:.2f} - {self.current_request} waited for {self.current_request.waiting_time:.2f} time units")
        
        # First, have the container scale its resources for the request
        scaling_result = self.scale_for_request()
        
        # If scaling failed, abort the service process
        if not scaling_result:
            print(f"{self.env.now:.2f} - ERROR: {self} failed to scale for {self.current_request}. Aborting service.")
            exit(1)  # Exit if we can't scale the container

        self.state = "Active"
        # Update request state to "Running" when container is active
        self.current_request.state = "Running"
        if variables.VERBOSE:
            print(f"{self.env.now:.2f} - {self.current_request} state changed to Running")

        # Determine service rate based on app type
        service_rate = variables.APPLICATIONS[self.current_request.app_id]["service_rate"]
            
        self.current_request.processing_time = random.expovariate(service_rate)*self.cluster.processing_time_factor
        # self.current_request.processing_time = service_time  # Track processing time for this request
        service_time = self.current_request.processing_time + self.current_request.network_delay
        if variables.VERBOSE:
            print(f"{self.env.now:.2f} - {self.current_request} starting service in {self}. Expected duration: {service_time:.2f}")
        yield self.env.timeout(service_time)
        if variables.VERBOSE:
            print(f"{self.env.now:.2f} - {self.current_request} completed service time.")

    def idle_lifecycle(self):
        """Manages the idle timeout for a container."""
        if variables.VERBOSE:
            print(f"{self.env.now:.2f} - {self} is now idle. Starting idle timeout.")
        try:
            # Use the scheduler to calculate the idle timeout
            # scheduler = self.system.schedulers[self.cluster.name]
            # idle_timeout = scheduler.calculate_idle_timeout(self)
            
            if variables.VERBOSE:
                print(f"{self.env.now:.2f} - Generated exponential idle timeout: {self.time_out:.2f}s for {self}")
            yield self.env.timeout(self.time_out)
            # If timeout completes without interruption, remove the container
            if variables.VERBOSE:
                print(f"{self.env.now:.2f} - Idle timeout reached for {self}. Removing it.")
            variables.request_stats['containers_removed_idle'] += 1
            variables.app_stats[self.app_id]['containers_removed_idle'] += 1
            
            self.state = "Dead"  # Mark as expired

            # Remove from server's list
            if self in self.server.containers:
                self.server.containers.remove(self)
            
            # Remove from idle container store in the system
            self.system.app_idle_containers[self.cluster.name][self.app_id].remove(self)

            # Mark container as dead and release resources
            self.release_resources()


            # We don't need to explicitly remove from idle_containers store here,
            # as it would have been removed by 'get' if reused. If it times out,
            # it's effectively unusable anyway after releasing resources.

        except simpy.Interrupt:
            # Interrupted means it was reused before timeout!
            if variables.VERBOSE:
                print(f"{self.env.now:.2f} - {self} reused before idle timeout. Interrupt received.")
            # No need to do anything here, the assign_request method handled the state change.