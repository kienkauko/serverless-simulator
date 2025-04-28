import itertools
import simpy 
import random
from variables import CONTAINER_ASSIGN_RATE
from variables import request_stats, app_stats, latency_stats, app_latency_stats, APPLICATIONS

class Container:
    """Represents a container instance spawned on a server."""
    # Using itertools for unique container IDs across the system
    id_counter = itertools.count()

    def __init__(self, env, system, server, cpu_alloc, ram_alloc, cpu_reserve, ram_reserve, app_id=None, cluster_name=None):
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
        self.app_id = app_id # Application ID this container belongs to
        self.cluster_name = cluster_name # Cluster this container belongs to

    def __str__(self):
        state = f"Serving {self.current_request.id}" if self.current_request else f"Idle since {self.idle_since:.2f}"
        app_info = f" [App: {self.app_id}]" if self.app_id else ""
        cluster_info = f" [Cluster: {self.cluster_name}]" if self.cluster_name else ""
        return f"Cont_{self.id}(on Srv_{self.server.id}, State: {state}){app_info}{cluster_info}"

    def scale_for_request(self):
        """Scales container resources for the current request."""
        if not self.current_request:
            print(f"ERROR: {self.env.now:.2f} - scale_for_request called for {self} with no request!")
            return False
            
        request = self.current_request
            
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
                # Reset the container state and clear the current_request since scaling failed
                self.state = "Idle"
                self.current_request = None
                exit(1)
            
            print(f"{self.env.now:.2f} - {self} allocated resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) for {request} on {self.server}")
            
            # Release the lock after successful allocation
            self.server.resource_lock.release(lock_request)
                
            # Update the container's allocated resources
            self.cpu_alloc = request.cpu_demand
            self.ram_alloc = request.ram_demand
            
            # Update the request's service start time
            request.start_service_time = self.env.now
                
            return True
        
        except Exception as e:
            # Make sure we release the lock even if there's an error
            print(f"{self.env.now:.2f} - ERROR in scale_for_request: {e}")
            self.server.resource_lock.release(lock_request)
            # Reset the container state and clear the current_request since scaling failed
            self.state = "Idle"
            self.current_request = None
            return False

    def release_request(self):
        """Releases the finished request and marks the container as idle."""
        if not self.current_request:
            print("abc")
            return # Nothing to release

# Use the server's resource lock to prevent race conditions
        lock_request = self.server.resource_lock.request()
        yield lock_request
        
        try:
        # Modify resources in the container
            delta_cpu = self.cpu_alloc - self.current_request.cpu_warm
            delta_ram = self.ram_alloc - self.current_request.ram_warm

            if(delta_cpu < 0 or delta_ram < 0):
                print(f"FATAL ERROR: {self.env.now:.2f} - {self} trying to release more resources than allocated (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f})")
                exit(1)

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
        self.server.resource_lock.release(lock_request)

        # Clear the current request
        finished_request = self.current_request
        finished_request.end_service_time = self.env.now
        print(f"{self.env.now:.2f} - {finished_request} finished service in {self}. Duration: {finished_request.end_service_time - finished_request.start_service_time:.2f}")
        print(f"{self.env.now:.2f} - {finished_request} releasing resources (CPU:{delta_cpu:.1f}, RAM:{delta_ram:.1f}) from {self.server}")
        
        self.current_request = None

    def release_resources(self):
        """Returns allocated CPU and RAM resources back to the server."""
        print(f"{self.env.now:.2f} - {self} releasing resources (CPU:{self.cpu_alloc:.1f}, RAM:{self.ram_alloc:.1f}) from {self.server}")
        
        # Use the server's resource lock to prevent race conditions
        lock_request = self.server.resource_lock.request()
        yield lock_request
        

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

        self.server.resource_lock.release(lock_request)


    def service_lifecycle(self, path, topology, use_topology):
        """Simulates the request processing time within a container."""
        # if not self.current_request:
        #     print(f"ERROR: {self.env.now:.2f} - service_lifecycle called for {self} with no request!")
        #     return
        request = self.current_request
        
        # Calculate and record the waiting time
        if request.waiting_start_time != -1:
            request.waiting_time = self.env.now - request.waiting_start_time
            # print(f"{self.env.now:.2f} - {request} waited for {request.waiting_time:.2f} time units")
        
        # First, have the container scale its resources for the request
        scaling_result = yield self.env.process(self.scale_for_request())
        
        # If scaling failed, abort the service process
        if not scaling_result:
            print(f"{self.env.now:.2f} - Failed to scale {self} for {request}. Aborting service.")
            # Release network resources if using topology
            if use_topology and path:
                bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
                topology.release_path(path, bandwidth_demand)
            exit(1)  # Exit if we can't scale the container

        self.state = "Active"

        # Determine service rate based on app type
        service_rate = APPLICATIONS[request.app_id]["service_rate"]
            
        service_time = random.expovariate(service_rate)
        print(f"{self.env.now:.2f} - {request} starting service in {self}. Expected duration: {service_time:.2f}")
        yield self.env.timeout(service_time)

        # Service finished
        request_stats['processed'] += 1
        app_stats[request.app_id]['processed'] += 1
            
        # Compute latencies: sum of propagation, spawn, and processing times
        processing_time = service_time
        total_latency = request.prop_delay + request.spawn_time + processing_time
        
        # Update global latency stats
        latency_stats['total_latency'] += total_latency
        latency_stats['propagation_delay'] += request.prop_delay
        latency_stats['spawning_time'] += request.spawn_time
        latency_stats['processing_time'] += processing_time
        latency_stats['waiting_time'] += request.waiting_time  # Add waiting time to stats
        latency_stats['count'] += 1
        
        # Update app-specific latency stats
        app_latency_stats[request.app_id]['total_latency'] += total_latency
        app_latency_stats[request.app_id]['propagation_delay'] += request.prop_delay
        app_latency_stats[request.app_id]['spawning_time'] += request.spawn_time
        app_latency_stats[request.app_id]['processing_time'] += processing_time
        app_latency_stats[request.app_id]['waiting_time'] += request.waiting_time  # Add app-specific waiting time
        app_latency_stats[request.app_id]['count'] += 1
            
        print(f"{self.env.now:.2f} - {request} latencies recorded: Total {total_latency:.2f}, "
              f"Propagation {request.prop_delay:.2f}, Spawn {request.spawn_time:.2f}, "
              f"Processing {processing_time:.2f}, Waiting {request.waiting_time:.2f}")

        # Release the request and mark the container as idle
        yield self.env.process(self.release_request())
        
        # Release the bandwidth once service is complete if using topology
        if use_topology and path:
            # Get app-specific bandwidth demand for release
            bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
            topology.release_path(path, bandwidth_demand)

        # Update the container's state
        self.state = "Idle"
        self.idle_since = self.env.now
        # Put the container into the idle pool
        self.system.add_idle_container(self, self.cluster_name)
        # Start the idle timeout process
        self.idle_timeout_process = self.env.process(self.idle_lifecycle())

    def idle_lifecycle(self):
        """Manages the idle timeout for a container."""
        print(f"{self.env.now:.2f} - {self} is now idle. Starting idle timeout.")
        try:
            # Use the scheduler to calculate the idle timeout
            scheduler = self.system.schedulers[self.cluster_name]
            idle_timeout = scheduler.calculate_idle_timeout(self)
            
            print(f"{self.env.now:.2f} - Generated exponential idle timeout: {idle_timeout:.2f}s for {self}")
            yield self.env.timeout(idle_timeout)
            # If timeout completes without interruption, remove the container
            print(f"{self.env.now:.2f} - Idle timeout reached for {self}. Removing it.")
            request_stats['containers_removed_idle'] += 1
            app_stats[self.app_id]['containers_removed_idle'] += 1
            
            # Remove from server's list
            if self in self.server.containers:
                self.server.containers.remove(self)
            
            self.state = "Dead"  # Mark as expired

            # Mark container as dead and release resources
            yield self.env.process(self.release_resources())


            # We don't need to explicitly remove from idle_containers store here,
            # as it would have been removed by 'get' if reused. If it times out,
            # it's effectively unusable anyway after releasing resources.

        except simpy.Interrupt:
            # Interrupted means it was reused before timeout!
            print(f"{self.env.now:.2f} - {self} reused before idle timeout. Interrupt received.")
            # No need to do anything here, the assign_request method handled the state change.