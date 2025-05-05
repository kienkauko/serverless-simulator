import simpy
import random
import itertools
from variables import *
from Request import Request
from Container import Container 

class System:
    """Orchestrates the simulation, managing servers, requests, and containers."""
    def __init__(self, env, config, distribution='exponential', verbose=False):
        self.env = env
        self.servers = []  # Direct list of servers
        self.idle_containers = simpy.Store(env)
        self.config = config
        self.arrival_rate = config["request"]["arrival_rate"]
        self.service_rate = config["request"]["service_rate"]
        self.spawn_time_mean = config["container"]["spawn_time"]
        self.container_idle_timeout = config["container"]["idle_timeout"]
        self.distribution = distribution  # Added parameter for distribution type
        self.verbose = verbose  # Flag to control logging output
        self.req_id_counter = itertools.count()
        # Variables for tracking waiting requests
        self.waiting_requests_count = 0
        self.last_waiting_update = 0.0
        self.total_waiting_area = 0.0  # Time-weighted area for Little's Law calculation
        self.request_pending = 0  # Number of requests pending in the system
        self.request_running = 0  # Number of requests currently being processed
        
    def add_server(self, server):
        """Add a server to the system"""
        self.servers.append(server)

    def request_generator(self):
        """Generates requests according to a Poisson process."""
        while True:
            # Time between arrivals (Exponential distribution for Poisson process)
            inter_arrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            # Generate request details
            req_id = next(self.req_id_counter)
            arrival_time = self.env.now
            
            # Get fixed CPU and RAM values from config
            cpu_warm = self.config["request"]["cpu_warm"]
            ram_warm = self.config["request"]["ram_warm"]
            cpu_demand = self.config["request"]["cpu_demand"]
            ram_demand = self.config["request"]["ram_demand"]
            
            request = Request(req_id, arrival_time, cpu_demand, ram_demand, cpu_warm, ram_warm)
            request_stats['generated'] += 1
            if self.verbose:
                print(f"{self.env.now:.2f} - Request Generated: {request}")
            # Start the handling process for this request
            self.env.process(self.handle_request(request))

    def handle_request(self, request):
        """Handles an incoming request: waits for an idle container from the pool."""
        
        # Record the time when the request starts waiting for a container
        request_wait_start_time = self.env.now
        self.increment_waiting()  # Track waiting requests
        
        # Check if there are any idle containers available
        idle_containers = sum(1 for container in self.idle_containers.items if container.state == "Idle")
        
        if idle_containers > 0:
            # Idle containers available, try to assign request directly
            if self.verbose:
                print(f"{self.env.now:.2f} - Found {idle_containers} idle containers, trying to assign request directly.")
            
            while True:
                get_op = self.idle_containers.get()
                container = yield get_op
                
                if container.state != "Idle":
                    if self.verbose:
                        print(f"WARNING: {self.env.now:.2f} - Retrieved container {container} is not idle, but {container.state}. Discarding it.")
                    continue
                    
                if self.verbose:
                    print(f"{self.env.now:.2f} - Found idle {container} for {request}")
                request_stats['containers_reused'] += 1
                
                # Record the time just before starting the assignment process
                # Start the assignment process and store its handle in the container
                container.assignment_process = self.env.process(container.assign_request(request))
                assignment_result = yield container.assignment_process
                container.assignment_process = None  # Clear the process handle after completion
                
                if not assignment_result:
                    print(f"{self.env.now:.2f} - FATAL ERROR: Assignment to {container} failed.")
                    exit(1)
                
                # Calculate total waiting time
                total_wait_time = self.env.now - request_wait_start_time
                if self.verbose:
                    print(f"{self.env.now:.2f} - Total wait time of request {request}: {total_wait_time:.2f})")
                
                # Store the waiting time in the request object
                request.wait_time = total_wait_time
                self.decrement_waiting()  # Request no longer waiting
                if self.verbose:
                    print(f"{self.env.now:.2f} - {request} waited {total_wait_time:.2f} time units")
                
                # Update the waiting time statistics
                latency_stats['waiting_time'] += total_wait_time
                
                self.env.process(self.container_service_lifecycle(container))
                return
        
        else:
            # No idle containers available, check if we can spawn a new one
            if self.verbose:
                print(f"{self.env.now:.2f} - No idle containers available. Checking resources for spawning.")
            server = self.find_server_for_spawn(request)
            
            if server:
                # Resources available, spawn new container for this request
                if self.verbose:
                    print(f"{self.env.now:.2f} - Resources available on {server}. Spawning container for request {request}.")
                request_stats['container_spawns_initiated'] += 1
                self.request_pending += 1  # Increment pending requests
                request.state = "Pending"  # Update request state to Pending
                
                # Spawn the container and wait for completion
                spawn_process = self.env.process(self.spawn_container_process(server, request))
                spawned_container = yield spawn_process
                
                if spawned_container:
                    # Container spawned successfully, assign the request directly
                    request_stats['container_spawns_succeeded'] += 1
                    
                    # Start the assignment process
                    spawned_container.assignment_process = self.env.process(spawned_container.assign_request(request))
                    assignment_result = yield spawned_container.assignment_process
                    spawned_container.assignment_process = None
                    
                    if assignment_result:
                        # Calculate waiting time
                        total_wait_time = self.env.now - request_wait_start_time
                        
                        # Update request status and counters
                        self.request_pending -= 1
                        request.wait_time = total_wait_time
                        self.decrement_waiting()
                        
                        if self.verbose:
                            print(f"{self.env.now:.2f} - {request} waited {total_wait_time:.2f} time units")
                        
                        # Update statistics
                        latency_stats['waiting_time'] += total_wait_time
                        
                        # Start the service process
                        self.env.process(self.container_service_lifecycle(spawned_container))
                        return
                    else:
                        print(f"{self.env.now:.2f} - ERROR: Assignment to newly spawned container failed.")
                        exit(1)
                else:
                    # Container spawn failed
                    print(f"{self.env.now:.2f} - FATAL ERROR: Container spawn failed on {server}.")
                    exit
            else:
                # No server has enough resources
                if self.verbose:
                    print(f"{self.env.now:.2f} - BLOCK: No server with sufficient capacity for {request}")
                request_stats['blocked_no_server_capacity'] += 1
                request.state = "Rejected"
                self.decrement_waiting()
                return

    def find_server_for_spawn(self, request):
        """Finds the first server with enough *current* capacity (First-Fit)."""
        for server in self.servers:
            if server.has_capacity(request.cpu_demand, request.ram_demand, request.cpu_warm, request.ram_warm):
                return server
        return None

    def spawn_container_process(self, server, request):
        """Simulates the time and resource acquisition for spawning a container."""
        try:
            # Request the lock to prevent race conditions
            lock_request = server.resource_lock.request()
            yield lock_request
            
            # Check available resources first before attempting to allocate
            if server.cpu_reserve < request.cpu_demand or server.ram_reserve < request.ram_demand or \
               server.cpu_real < request.cpu_warm or server.ram_real < request.ram_warm:
                print(f"{self.env.now:.2f} - Resource check failed just before acquiring on {server} for {request}.")
                server.resource_lock.release(lock_request)
                exit(1)

            # Allocate resources directly from server variables
            server.cpu_real -= request.cpu_warm
            server.ram_real -= request.ram_warm
            server.cpu_reserve -= request.cpu_demand
            server.ram_reserve -= request.ram_demand

            if server.cpu_real < 0 or server.ram_real < 0 or server.cpu_reserve < 0 or server.ram_reserve < 0:
                print(f"{self.env.now:.2f} - ERROR: Resource allocation failed on {server} for {request}.")
                exit(1)
            
            # Release the lock after allocation
            server.resource_lock.release(lock_request)

            # Determine spawn time based on distribution type
            if self.distribution == 'exponential':
                # Exponentially distributed spawn time
                spawn_time_real = random.expovariate(1.0/self.spawn_time_mean)
            else:
                # Deterministic spawn time equal to the mean
                spawn_time_real = self.spawn_time_mean
                
            if self.verbose:
                print(f"{self.env.now:.2f} - Waiting {spawn_time_real:.2f} time units for spawning...")
            yield self.env.timeout(spawn_time_real)

            # Record the spawn time for latency calculation
            request.spawn_time = spawn_time_real

            # Spawning complete, create the container object
            container = Container(self.env, self, server, request.cpu_warm, request.ram_warm, request.cpu_demand, request.ram_demand)
            server.containers.append(container) # Track container on server
            if self.verbose:
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

    def container_service_lifecycle(self, container):
        """Simulates the request processing time within a container."""
        if not container.current_request:
            print(f"ERROR: {self.env.now:.2f} - container_service_lifecycle called for {container} with no request!")
            return

        request = container.current_request
        service_time = random.expovariate(self.service_rate)
        if self.verbose:
            print(f"{self.env.now:.2f} - {request} starting service in {container}. Expected duration: {service_time:.2f}")
        # Update the request's state to "Running" when service starts
        request.state = "Running"
        self.request_running += 1  # Increment the count of requests in the system

        yield self.env.timeout(service_time)

        # Service finished
        request_stats['processed'] += 1
        # Update the request's state to "Finished" when service completes
        request.state = "Finished"
        container.release_request()
        self.request_running -= 1  # Decrement the count of requests in the system
        
        # Compute latencies: sum of spawn, waiting, and processing times
        processing_time = service_time
        total_latency = request.wait_time + processing_time
        
        # Update global latency stats
        latency_stats['total_latency'] += total_latency
        latency_stats['spawning_time'] += request.spawn_time
        latency_stats['processing_time'] += processing_time
        latency_stats['count'] += 1
        
        if self.verbose:
            print(f"{self.env.now:.2f} - {request} latencies recorded: Total {total_latency:.2f}, "
                  f"Spawn {request.spawn_time:.2f}, "
                  f"Wait {request.wait_time:.2f}, Processing {processing_time:.2f}")

    def container_idle_lifecycle(self, container):
        """Manages the idle timeout for a container."""
        try:
            # Determine idle timeout based on distribution type
            if self.distribution == 'exponential':
                # Exponentially distributed idle timeout
                idle_timeout = random.expovariate(1.0/self.container_idle_timeout)
            else:
                # Deterministic idle timeout equal to the mean
                idle_timeout = self.container_idle_timeout
                
            if self.verbose:
                print(f"{self.env.now:.2f} - Generated idle timeout: {idle_timeout:.2f}s for {container}")
            yield self.env.timeout(idle_timeout)
            
            # If timeout completes without interruption, remove the container
            if self.verbose:
                print(f"{self.env.now:.2f} - Idle timeout reached for {container}. Removing it.")
            request_stats['containers_removed_idle'] += 1

            # Check if there's an active assignment process and interrupt it
            if container.assignment_process and not container.assignment_process.triggered:
                if self.verbose:
                    print(f"{self.env.now:.2f} - Interrupting ongoing assignment process for {container}")
                container.assignment_process.interrupt()
            
            # Mark container as dead and release resources
            container.state = "Dead"  # Mark as expired
            
            container.release_resources()

        except simpy.Interrupt:
            # Interrupted means it was reused before timeout!
            if self.verbose:
                print(f"{self.env.now:.2f} - {container} reused before idle timeout. Interrupt received.")

    def add_idle_container(self, container):
        """Adds a container to the idle store."""
        if self.verbose:
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