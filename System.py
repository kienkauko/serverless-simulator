import simpy
import random
import itertools
from variables import *
from Request import Request
from Container import Container 
from Topology import Topology

# --- System Class ---

class System:
    """Orchestrates the simulation, managing servers, requests, and containers."""
    def __init__(self, env, arrival_rate, service_rate, spawn_time, idle_timeout, topology, cluster, use_topology=True):
        self.env = env
        self.topology = topology  # New: topology instance
        self.cluster = cluster    # New: cluster instance (holds its servers)
        self.idle_containers = simpy.Store(env)
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.spawn_time_mean = spawn_time
        self.container_idle_timeout = idle_timeout
        self.req_id_counter = itertools.count()
        self.use_topology = use_topology  # Flag to control topology usage
        
        # Add these new variables for tracking waiting requests
        self.waiting_requests_count = 0
        self.last_waiting_update = 0.0
        self.total_waiting_area = 0.0  # Time-weighted area for Little's Law calculation

    def request_generator(self):
        """Generates requests according to a Poisson process."""
        while True:
            # Time between arrivals (Exponential distribution for Poisson process)
            inter_arrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            # Generate request details
            req_id = next(self.req_id_counter)
            arrival_time = self.env.now
            
            # For topology-aware simulations, assign a random node from the topology
            origin = None
            if self.use_topology:
                origin = random.choice(list(self.topology.graph.nodes))
                
            request = Request(req_id, arrival_time, CPU_DEMAND, RAM_DEMAND, CPU_WARM, RAM_WARM, origin_node=origin)
            request_stats['generated'] += 1
            print(f"{self.env.now:.2f} - Request Generated: {request}")

            # Start the handling process for this request
            self.env.process(self.handle_request(request))

    def handle_request(self, request):
        """Handles an incoming request: waits for an idle container from the pool."""
        
        path = None
        # Record the time when the request starts waiting for a container
        request_wait_start_time = self.env.now
        self.increment_waiting()  # Add this line to track waiting requests
        
        # Handle topology routing if enabled
        if self.use_topology:
            # First, try to route from request.origin_node to the cluster node.
            path = self.topology.find_path(request.origin_node, self.cluster.node, BANDWIDTH_DEMAND)
            if not path:
                print(f"{self.env.now:.2f} - BLOCK: No feasible path from {request.origin_node} to {self.cluster.node} for {request}")
                request_stats['blocked_no_path'] += 1
                self.decrement_waiting()  # Add this
                return

            # Compute round-trip propagation delay (sum of latencies *2)
            prop = sum(self.topology.graph.get_edge_data(path[i], path[i+1])['latency'] for i in range(len(path)-1))
            request.prop_delay = 2 * prop
        else:
            # No topology routing - set propagation delay to 0
            request.prop_delay = 0.0

        # Check if we need to attempt spawning a new container
        idle_containers = sum(1 for container in self.idle_containers.items if container.state == "Idle")
        if idle_containers == 0:
            # No idle containers available, try to spawn a new one
            print(f"{self.env.now:.2f} - No idle container found. Attempting to spawn for future use.")
            server = self.find_server_for_spawn(request)
            if server:
                # print(f"{self.env.now:.2f} - Found potential {server} for spawning container")
                request_stats['container_spawns_initiated'] += 1
                self.env.process(self.spawn_container_to_idle_pool(server, request))
            else:
                print(f"{self.env.now:.2f} - BLOCK: No idle containers and no server with sufficient capacity for {request}")
                request_stats['blocked_no_server_capacity'] += 1
                self.decrement_waiting()  # Add this
                
                # Release the path if we're using topology and have already allocated one
                if self.use_topology and path:
                    self.topology.release_path(path, BANDWIDTH_DEMAND)
                
                # Block this request by returning from the method (not continuing to wait)
                return
        
        # Wait for an idle container (whether pre-existing or newly spawned)
        # print(f"{self.env.now:.2f} - {request} waiting for an idle container")
        while True:
            get_op = self.idle_containers.get()
            container = yield get_op
            
            if container.state != "Idle":
                print(f"WARNING: {self.env.now:.2f} - Retrieved container {container} is not idle, but {container.state}. Discarding it.")
                continue
                
            print(f"{self.env.now:.2f} - Found idle {container} for {request}")
            request_stats['containers_reused'] += 1
            
            # Record the time just before starting the assignment process
            pre_assignment_time = self.env.now
            
            # Start the assignment process and store its handle in the container
            container.assignment_process = self.env.process(container.assign_request(request))
            assignment_result = yield container.assignment_process
            container.assignment_process = None  # Clear the process handle after completion
            
            if not assignment_result:
                continue
            # Calculate total waiting time: wait for container + assignment time
            post_assignment_time = self.env.now
            assignment_time = post_assignment_time - pre_assignment_time
            total_wait_time = post_assignment_time - request_wait_start_time
            print(f"{self.env.now:.2f} - Total wait time of request {request}: {total_wait_time:.2f})")
            # If assignment was successful, process the service
            if assignment_result:
                # Store the waiting time in the request object
                request.wait_time = total_wait_time
                self.decrement_waiting()  # Add this line when a request is no longer waiting
                print(f"{self.env.now:.2f} - {request} waited {total_wait_time:.2f} time units (container wait: {pre_assignment_time - request_wait_start_time:.2f}, assignment: {assignment_time:.2f})")
                
                # Update the waiting time statistics
                latency_stats['waiting_time'] += total_wait_time
                latency_stats['container_wait_time'] += (pre_assignment_time - request_wait_start_time)
                latency_stats['assignment_time'] += assignment_time
                
                self.env.process(self.container_service_lifecycle(container, path))
                return
            else:
                # If assignment failed due to resource constraints, try another container
                print(f"{self.env.now:.2f} - Assignment to {container} failed. Trying another container.")
                request_stats['reuse_oom_failures'] += 1
                continue

    def spawn_container_to_idle_pool(self, server, request):
        """Spawns a container and adds it directly to the idle pool."""
        spawn_process = self.env.process(self.spawn_container_process(server, request))
        spawned_container = yield spawn_process
        
        if spawned_container:
            # Add the container to the idle pool immediately
            request_stats['container_spawns_succeeded'] += 1
            print(f"{self.env.now:.2f} - Successfully spawned {spawned_container}. Adding to idle pool.")
            self.add_idle_container(spawned_container)
        else:
            print(f"{self.env.now:.2f} - Spawn failed on {server}.")
            request_stats['container_spawns_failed'] += 1

    def find_server_for_spawn(self, request):
        """Finds the first server with enough *current* capacity (First-Fit)."""
        for server in self.cluster.servers:
            if server.has_capacity(request.cpu_demand, request.ram_demand, request.cpu_warm, request.ram_warm):
                return server
        return None

    def spawn_container_process(self, server, request):
        """Simulates the time and resource acquisition for spawning a container."""
        # print(f"{self.env.now:.2f} - Attempting to acquire resources on {server} for {request}...")

        try:
            # Request the lock to prevent race conditions
            lock_request = server.resource_lock.request()
            yield lock_request
            
            # Check available resources first before attempting to allocate
            if server.cpu_reserve < request.cpu_demand or server.ram_reserve < request.ram_demand or \
               server.cpu_real < request.cpu_warm or server.ram_real < request.ram_warm:
                print(f"{self.env.now:.2f} - Resource check failed just before acquiring on {server} for {request}.")
                server.resource_lock.release(lock_request)
                self.env.exit(None)  # Signal spawn failure

            # Log levels before acquiring resources
            # print(f"{self.env.now:.2f} - Server before spawn: {server}")
            # print(f"{self.env.now:.2f} - Demanding resources: CPU_Real {request.cpu_warm:.1f}, RAM_Real {request.ram_warm:.1f}, CPU_Reserve {request.cpu_demand:.1f}, RAM_Reserve {request.ram_demand:.1f}")

            # Allocate resources directly from server variables
            server.cpu_real -= request.cpu_warm
            server.ram_real -= request.ram_warm
            server.cpu_reserve -= request.cpu_demand
            server.ram_reserve -= request.ram_demand

            # Log levels after acquiring resources
            # print(f"{self.env.now:.2f} - Server after spawn: {server}")
            
            # Release the lock after allocation
            server.resource_lock.release(lock_request)

            # Resources acquired, now wait for spawn time, which is exponential distributed
            spawn_time_real = random.expovariate(1.0/self.spawn_time_mean)
            print(f"Waiting {spawn_time_real:.2f} time units for spawning...")
            yield self.env.timeout(spawn_time_real)

            # Record the spawn time for latency calculation
            request.spawn_time = spawn_time_real

            # Spawning complete, create the container object
            container = Container(self.env, self, server, request.cpu_warm, request.ram_warm, request.cpu_demand, request.ram_demand)
            server.containers.append(container) # Track container on server
            print(f"{self.env.now:.2f} - Spawn Complete: Created {container}")
            
            # Start the idle timeout process immediately after container creation
            container.idle_since = self.env.now
            container.idle_timeout_process = self.env.process(self.container_idle_lifecycle(container))
            
            return container  # Return the created container object

        except Exception as e:  # Catch other potential issues
            print(f"ERROR: {self.env.now:.2f} - Unexpected error during spawn for {request} on {server}: {e}")
            # Make sure we release the lock if we have it and there's an error
            if 'lock_request' in locals():
                try:
                    server.resource_lock.release(lock_request)
                except:
                    pass
            self.env.exit(None)  # Signal spawn failure

    def container_service_lifecycle(self, container, path):
        """Simulates the request processing time within a container."""
        if not container.current_request:
            print(f"ERROR: {self.env.now:.2f} - container_service_lifecycle called for {container} with no request!")
            return

        request = container.current_request
        service_time = random.expovariate(self.service_rate)
        print(f"{self.env.now:.2f} - {request} starting service in {container}. Expected duration: {service_time:.2f}")
        yield self.env.timeout(service_time)

        # Service finished
        request_stats['processed'] += 1
        container.release_request()
        
        # Release the bandwidth once service is complete if using topology
        if self.use_topology and path:
            self.topology.release_path(path, BANDWIDTH_DEMAND)
            
        # Compute latencies: sum of propagation, spawn, and processing times
        processing_time = service_time
        total_latency = request.prop_delay + request.wait_time + processing_time
        
        # Update global latency stats
        latency_stats['total_latency'] += total_latency
        latency_stats['propagation_delay'] += request.prop_delay
        latency_stats['spawning_time'] += request.spawn_time
        latency_stats['processing_time'] += processing_time
        latency_stats['count'] += 1
        
        print(f"{self.env.now:.2f} - {request} latencies recorded: Total {total_latency:.2f}, "
              f"Propagation {request.prop_delay:.2f}, Spawn {request.spawn_time:.2f}, "
              f"Wait {request.wait_time:.2f}, Processing {processing_time:.2f}")

    def container_idle_lifecycle(self, container):
        """Manages the idle timeout for a container."""
        # print(f"{self.env.now:.2f} - {container} is now idle. Starting idle timeout ({self.container_idle_timeout}s).")
        try:
            # Generate exponentially distributed idle timeout with mean = self.container_idle_timeout
            idle_timeout = random.expovariate(1.0/self.container_idle_timeout)
            print(f"{self.env.now:.2f} - Generated exponential idle timeout: {idle_timeout:.2f}s for {container}")
            yield self.env.timeout(idle_timeout)
            # If timeout completes without interruption, remove the container
            print(f"{self.env.now:.2f} - Idle timeout reached for {container}. Removing it.")
            request_stats['containers_removed_idle'] += 1

            # Check if there's an active assignment process and interrupt it
            if container.assignment_process and not container.assignment_process.triggered:
                print(f"{self.env.now:.2f} - Interrupting ongoing assignment process for {container}")
                container.assignment_process.interrupt()
            
            # Mark container as dead and release resources
            container.state = "Dead"  # Mark as expired
            container.release_resources()
            # We don't need to explicitly remove from idle_containers store here,
            # as it would have been removed by 'get' if reused. If it times out,
            # it's effectively unusable anyway after releasing resources.

        except simpy.Interrupt:
            # Interrupted means it was reused before timeout!
            print(f"{self.env.now:.2f} - {container} reused before idle timeout. Interrupt received.")
            # No need to do anything here, the assign_request method handled the state change.

    def add_idle_container(self, container):
        """Adds a container to the idle store."""
        print(f"{self.env.now:.2f} - Adding {container} to idle pool.")
        self.idle_containers.put(container)

    def update_waiting_stats(self):
        """Update time-weighted statistics for waiting requests."""
        current_time = self.env.now
        time_delta = current_time - self.last_waiting_update
        self.total_waiting_area += time_delta * self.waiting_requests_count
        self.last_waiting_update = current_time

    def increment_waiting(self):
        """Increment the count of waiting requests."""
        self.update_waiting_stats()
        self.waiting_requests_count += 1

    def decrement_waiting(self):
        """Decrement the count of waiting requests."""
        self.update_waiting_stats()
        self.waiting_requests_count -= 1