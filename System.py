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
        """Handles an incoming request: finds an idle container or spawns a new one."""
        
        path = None
        # Handle topology routing if enabled
        if self.use_topology:
            # First, try to route from request.origin_node to the cluster node.
            path = self.topology.find_path(request.origin_node, self.cluster.node, BANDWIDTH_DEMAND)
            if not path:
                print(f"{self.env.now:.2f} - BLOCK: No feasible path from {request.origin_node} to {self.cluster.node} for {request}")
                request_stats['blocked_no_path'] += 1
                return

            # Compute round-trip propagation delay (sum of latencies *2)
            prop = sum(self.topology.graph.get_edge_data(path[i], path[i+1])['latency'] for i in range(len(path)-1))
            request.prop_delay = 2 * prop
            # print(f"{self.env.now:.2f} - Routed {request} via path: {path} (Propagation delay: {request.prop_delay:.2f})")
        else:
            # No topology routing - set propagation delay to 0
            request.prop_delay = 0.0

        # Use idle container if available
        while len(self.idle_containers.items) > 0:
            get_op = self.idle_containers.get()
            container = yield get_op
            if container.state != "Idle":
                print(f"WARNING: {self.env.now:.2f} - Retrieved container {container} is not idle. Discarding it.")
                continue

            print(f"{self.env.now:.2f} - Found idle {container} for {request}")
            request_stats['containers_reused'] += 1
            assignment_result = yield self.env.process(container.assign_request(request))
            
            # If assignment was successful, process the service
            if assignment_result:
                self.env.process(self.container_service_lifecycle(container, path))
                return
            else:
                # If assignment failed due to resource constraints, try another container
                print(f"{self.env.now:.2f} - Assignment to {container} failed. Trying another container.")
                request_stats['reuse_oom_failures'] += 1
                continue

        # No idle container; spawn a new one using a server from the cluster.
        print(f"{self.env.now:.2f} - No idle container found for {request}. Attempting to spawn.")
        server = self.find_server_for_spawn(request)
        if server:
            print(f"{self.env.now:.2f} - Found potential {server} for spawning container for {request}")
            request_stats['container_spawns_initiated'] += 1
            spawn_process = self.env.process(self.spawn_container_process(server, request))
            yield spawn_process
            spawned_container = spawn_process.value

            if spawned_container:
                # Record spawn time into request (spawn time already set inside spawn process)
                request_stats['container_spawns_succeeded'] += 1
                yield self.env.process(spawned_container.assign_request(request))
                self.env.process(self.container_service_lifecycle(spawned_container, path))
            else:
                print(f"{self.env.now:.2f} - BLOCK: Spawn failed for {request} on {server}.")
                request_stats['blocked_spawn_failed'] += 1
                request_stats['container_spawns_failed'] += 1
                if self.use_topology and path:
                    self.topology.release_path(path, BANDWIDTH_DEMAND)
        else:
            print(f"{self.env.now:.2f} - BLOCK: No server with sufficient capacity for {request}")
            request_stats['blocked_no_server_capacity'] += 1
            if self.use_topology and path:
                self.topology.release_path(path, BANDWIDTH_DEMAND)

    def find_server_for_spawn(self, request):
        """Finds the first server with enough *current* capacity (First-Fit)."""
        for server in self.cluster.servers:
            if server.has_capacity(request.cpu_demand, request.ram_demand, request.cpu_warm, request.ram_warm):
                return server
        return None

    def spawn_container_process(self, server, request):
        """Simulates the time and resource acquisition for spawning a container."""
        print(f"{self.env.now:.2f} - Attempting to acquire resources on {server} for {request}...")

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
            print(f"{self.env.now:.2f} - Server before spawn: {server}")
            print(f"{self.env.now:.2f} - Demanding resources: CPU_Real {request.cpu_warm:.1f}, RAM_Real {request.ram_warm:.1f}, CPU_Reserve {request.cpu_demand:.1f}, RAM_Reserve {request.ram_demand:.1f}")

            # Allocate resources directly from server variables
            server.cpu_real -= request.cpu_warm
            server.ram_real -= request.ram_warm
            server.cpu_reserve -= request.cpu_demand
            server.ram_reserve -= request.ram_demand

            # Log levels after acquiring resources
            print(f"{self.env.now:.2f} - Server after spawn: {server}")
            
            # Release the lock after allocation
            server.resource_lock.release(lock_request)

            # Resources acquired, now wait for spawn time, which is exponential distributed
            spawn_time_real = random.expovariate(1.0/self.spawn_time_mean)
            print(f"Waiting {spawn_time_real:.2f} time units...")
            yield self.env.timeout(spawn_time_real)

            # Record the spawn time for latency calculation
            request.spawn_time = spawn_time_real

            # Spawning complete, create the container object
            container = Container(self.env, self, server, request.cpu_warm, request.ram_warm, request.cpu_demand, request.ram_demand)
            server.containers.append(container) # Track container on server
            print(f"{self.env.now:.2f} - Spawn Complete: Created {container}")
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
        total_latency = request.prop_delay + request.spawn_time + processing_time
        # Update global latency stats
        latency_stats['total_latency'] += total_latency
        latency_stats['propagation_delay'] += request.prop_delay
        latency_stats['spawning_time'] += request.spawn_time
        latency_stats['processing_time'] += processing_time
        latency_stats['count'] += 1
        print(f"{self.env.now:.2f} - {request} latencies recorded: Total {total_latency:.2f}, "
              f"Propagation {request.prop_delay:.2f}, Spawn {request.spawn_time:.2f}, Processing {processing_time:.2f}")

    def container_idle_lifecycle(self, container):
        """Manages the idle timeout for a container."""
        print(f"{self.env.now:.2f} - {container} is now idle. Starting idle timeout ({self.container_idle_timeout}s).")
        try:
            # Generate exponentially distributed idle timeout with mean = self.container_idle_timeout
            idle_timeout = random.expovariate(1.0/self.container_idle_timeout)
            print(f"{self.env.now:.2f} - Generated exponential idle timeout: {idle_timeout:.2f}s for {container}")
            yield self.env.timeout(idle_timeout)
            # If timeout completes without interruption, remove the container
            print(f"{self.env.now:.2f} - Idle timeout reached for {container}. Removing it.")
            request_stats['containers_removed_idle'] += 1
            # Need to remove it from the idle store *if* it's still there
            # This is tricky because another process might be about to 'get' it.
            # A safer way is to let the 'get' succeed, but have the container mark itself as 'expired'.
            # For simplicity here, we'll just release resources. The store might briefly hold a dead ref.
            # If the container was successfully removed from store by `handle_request`, this does nothing bad.
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