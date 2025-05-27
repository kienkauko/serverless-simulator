import simpy
import random
import itertools
import math  # Added for sine calculations in day-night pattern
from variables import *
from Request import Request
from Container import Container 

class System:
    """Orchestrates the simulation, managing servers, requests, and containers."""
    def __init__(self, env, config, distribution='exponential', pattern_type='poisson', verbose=False):
        self.env = env
        self.servers = []  # Direct list of servers
        # self.idle_containers = simpy.Store(env)
        self.idle_model_containers = simpy.Store(env)
        self.idle_cpu_containers = simpy.Store(env)
        self.config = config
        self.arrival_rate = config["request"]["arrival_rate"]
        self.service_rate = config["request"]["service_rate"]
        self.spawn_container_time_mean = config["container"]["spawn_time"]
        self.load_request_time_mean = config["container"]["load_request_time"]
        self.load_model_time_mean = config["container"]["load_model_time"]
        self.container_idle_model_timeout = config["container"]["idle_model_timeout"]
        self.container_idle_cpu_timeout = config["container"]["idle_cpu_timeout"]
        self.distribution = distribution  # Added parameter for distribution type
        self.pattern_type = pattern_type  # Traffic pattern type: 'poisson' or 'day_night'
        self.verbose = verbose  # Flag to control logging output
        self.req_id_counter = itertools.count()
        
        # Day-night pattern parameters
        self.day_length = 24.0  # Length of a day in simulation time units
        self.peak_factor = 2.0  # Peak traffic is 2x the configured arrival rate
        self.trough_factor = 0.5  # Trough traffic is 0.5x the configured arrival rate
        self.peak_hour = 12.0  # Peak occurs at noon in simulation time
        
        # Variables for tracking waiting requests
        self.waiting_requests_count = 0
        self.last_waiting_update = 0.0
        self.total_waiting_area = 0.0  # Time-weighted area for Little's Law calculation
        self.request_pending = 0  # Number of requests pending in the system
        self.request_running = 0  # Number of requests currently being processed
        self.last_processing_update = 0.0
        self.total_processing_area = 0.0  # Time-weighted area for processing requests
        
        # Variables for tracking resource usage
        self.last_resource_update = 0.0
        self.total_cpu_usage_area = 0.0  # Time-weighted CPU usage
        self.total_ram_usage_area = 0.0  # Time-weighted RAM usage
        self.total_cpu_capacity = 0.0  # Total CPU capacity of all servers
        self.total_ram_capacity = 0.0  # Total RAM capacity of all servers
        
        # New variable to track detailed request information
        self.request_records = []  # List to store detailed information about each request
        
        # New variable to track resource consumption over time
        self.resource_history = []  # List to store snapshots of resource usage
        self.last_resource_snapshot = 0.0  # Time of the last resource snapshot
    
    def add_server(self, server):
        """Add a server to the system"""
        self.servers.append(server)
        # Update total system capacity
        self.total_cpu_capacity += server.cpu_capacity
        self.total_ram_capacity += server.ram_capacity

    def request_generator(self):
        """Generates requests according to a specified arrival pattern.
        
        Supports multiple pattern types:
        - 'poisson': Standard Poisson process with fixed arrival rate
        - 'day_night': Time-varying arrival rate based on day-night cycle
        - 'up_down': Traffic pattern that linearly increases to a peak and then decreases
        """
        while True:
            # Get the appropriate arrival rate based on pattern type
            if self.pattern_type == 'day_night':
                current_arrival_rate = self.calculate_day_night_arrival_rate()
            elif self.pattern_type == 'up_down':
                current_arrival_rate = self.calculate_up_down_arrival_rate()
            else:  # default to poisson
                current_arrival_rate = self.arrival_rate
                
            # Calculate inter-arrival time using the determined rate
            inter_arrival_time = random.expovariate(current_arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            # Generate request details
            req_id = next(self.req_id_counter)
            arrival_time = self.env.now
            
            # Get fixed CPU and RAM values from config
            resource_demand = self.config["request"]
            request = Request(req_id, arrival_time, resource_demand)
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
        
        # Try to reuse idle containers: model‐first, then CPU
        # See if there's any Idle_model container
        if any(c.state == "Idle_model" for c in self.idle_model_containers.items):
            store = self.idle_model_containers
            state_label = "Idle_model"
        # Otherwise, see if there's any Idle_cpu container
        elif any(c.state == "Idle_cpu" for c in self.idle_cpu_containers.items):
            store = self.idle_cpu_containers
            state_label = "Idle_cpu"
        else:
            store = None

        if store:
            if self.verbose:
                print(f"{self.env.now:.2f} - Found idle {state_label} containers, trying to assign request directly.")
            # Pull containers until we find one in the right idle state
            while True:
                container = yield store.get()
                if container.state != state_label:
                    if self.verbose:
                        print(f"{self.env.now:.2f} - WARNING: Retrieved {container} ({container.state}), discarding.")
                    continue
                # assign the request
                request_stats['containers_reused'] += 1
                container.assignment_process = self.env.process(container.assign_request(request))
                assignment_ok = yield container.assignment_process
                container.assignment_process = None
                if not assignment_ok:
                    print(f"{self.env.now:.2f} - FATAL ERROR: Assignment to {container} failed.")
                    exit(1)
                # record wait time
                total_wait = self.env.now - request_wait_start_time
                request.wait_time = total_wait
                self.decrement_waiting()
                # Only update latency statistics when system is stable
                if self.env.now > 0:
                    latency_stats['waiting_time'] += total_wait
                if self.verbose:
                    print(f"{self.env.now:.2f} - {request} waited {total_wait:.2f}")
                # start service
                self.env.process(self.container_service_lifecycle(container))
                return

        # if no idle_model or idle_cpu containers found, execution will fall through
        # to the “spawn new container” branch below
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
                        if self.env.now > 0:
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
                    exit(1)
            else:
                # No server has enough resources
                if self.verbose:
                    print(f"{self.env.now:.2f} - BLOCK: No server with sufficient capacity for {request}")
                request_stats['blocked_no_server_capacity'] += 1
                request.state = "Rejected"
                self.decrement_waiting()
                
                # Record rejected request information
                self.record_request_info(request)
                return

    def find_server_for_spawn(self, request):
        """Finds the first server with enough *current* capacity (First-Fit)."""
        for server in self.servers:
            if server.has_capacity(request.resource_info):
                return server
        return None

    def spawn_container_process(self, server, request):
        """Simulates the time and resource acquisition for spawning a container."""
        try:
            # Request the lock to prevent race conditions
            # lock_request = server.resource_lock.request()
            # yield lock_request
            # Update resource stats before allocation
            self.update_resource_stats()
            # Allocate resources directly from server variables
            server.cpu_real -= request.cpu_warm
            server.ram_real -= request.ram_warm
            server.cpu_reserve -= request.cpu_demand
            server.ram_reserve -= request.ram_demand

            if server.cpu_real < 0 or server.ram_real < 0 or server.cpu_reserve < 0 or server.ram_reserve < 0:
                print(f"{self.env.now:.2f} - ERROR: Resource allocation failed on {server} for {request}.")
                exit(1)
            
            # Release the lock after allocation
            # server.resource_lock.release(lock_request)

            # Wait for the container to be spawned
            if self.distribution == 'exponential':
                # Exponentially distributed spawn time
                spawn_time_real = random.expovariate(1.0/self.spawn_container_time_mean)
            else:
                # Deterministic spawn time equal to the mean
                spawn_time_real = self.spawn_container_time_mean
            if self.verbose:
                print(f"{self.env.now:.2f} - Waiting {spawn_time_real:.2f} time units for container spawning...")
            yield self.env.timeout(spawn_time_real)

            # Spawning complete, create the container object
            container = Container(self.env, self, server, request.resource_info)
            server.containers.append(container) # Track container on server
            if self.verbose:
                print(f"{self.env.now:.2f} - Spawn Complete: Created {container}")
            
             # Record the spawn time for latency calculation
            request.spawn_time += spawn_time_real
            # Allocate model resources
            yield self.env.process(container.assign_model(request))
           
            # Record the spawn time for latency calculation
            request.spawn_time += spawn_time_real

            return container  # Return the created container object

        except Exception as e:  # Catch other potential issues
            print(f"ERROR: {self.env.now:.2f} - Unexpected error during spawn for {request} on {server}: {e}")
            # Make sure we release the lock if we have it and there's an error
            # if 'lock_request' in locals():
            #     try:
            #         server.resource_lock.release(lock_request)
            #     except:
            #         pass
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
        self.increment_processing()  # Using the new method to track processing time

        yield self.env.timeout(service_time)

        # Record end service time for request
        request.end_service_time = self.env.now
        
        # Service finished
        request_stats['processed'] += 1
        # Update the request's state to "Finished" when service completes
        request.state = "Finished"
        
        container.release_request()
        self.decrement_processing()  # Using the new method to track processing time
        
        # Compute latencies: sum of spawn, waiting, and processing times
        processing_time = request.end_service_time - request.start_service_time
        total_latency = request.wait_time + processing_time
        
        # Update global latency stats
        if self.env.now > 0:
            latency_stats['total_latency'] += total_latency
            latency_stats['spawning_time'] += request.spawn_time
            latency_stats['processing_time'] += processing_time
            latency_stats['count'] += 1
            
        self.record_request_info(request)
        
        if self.verbose:
            print(f"{self.env.now:.2f} - {request} latencies recorded: Total {total_latency:.2f}, "
                  f"Spawn {request.spawn_time:.2f}, "
                  f"Wait {request.wait_time:.2f}, Processing {processing_time:.2f}")
        
        # Record successful request information

    def container_idle_lifecycle(self, container):
        """Manages the idle timeout for a container."""
        try:
            # Determine idle timeout based on distribution type
            if self.distribution == 'exponential':
                # Exponentially distributed idle timeout
                idle_timeout = random.expovariate(1.0/self.container_idle_model_timeout)
            else:
                # Deterministic idle timeout equal to the mean
                # idle_timeout = random.expovariate(1.0/self.container_idle_timeout)
                idle_timeout = self.container_idle_model_timeout
                
            if self.verbose:
                print(f"{self.env.now:.2f} - Start model idle timeout, duration: {idle_timeout:.2f}s for {container}")
            yield self.env.timeout(idle_timeout)
            
            # If timeout completes without interruption, continue the idle_cpu_timeout
            container.state = "Idle_cpu"  # Mark as idle CPU

            # Release model resources
            container.release_model()

            if self.verbose:
                print(f"{self.env.now:.2f} - Model resources released for {container}. Starting CPU idle timeout.")
            
            if self.distribution == 'exponential':
                # Exponentially distributed idle timeout
                idle_timeout = random.expovariate(1.0/self.container_idle_cpu_timeout)
            else:
                # Deterministic idle timeout equal to the mean
                # idle_timeout = random.expovariate(1.0/self.container_idle_timeout)
                idle_timeout = self.container_idle_cpu_timeout
                
            if self.verbose:
                print(f"{self.env.now:.2f} - Start CPU idle timeout, duration: {idle_timeout:.2f}s for {container}")
            yield self.env.timeout(idle_timeout)

            if self.verbose:
                print(f"{self.env.now:.2f} - Idle timeout reached for {container}. Removing it.")
            
            request_stats['containers_removed_idle'] += 1

            # Check if there's an active assignment process and interrupt it
            if container.assignment_process and not container.assignment_process.triggered:
                print(f"{self.env.now:.2f} - FATAL ERROR: Assignment process not triggered for {container}.")
                exit(1)
                # if self.verbose:
                #     print(f"{self.env.now:.2f} - Interrupting ongoing assignment process for {container}")
                # container.assignment_process.interrupt()
            
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
        if container.state == "Idle_model":
            self.idle_model_containers.put(container)
        elif container.state == "Idle_cpu":
            self.idle_cpu_containers.put(container)
        else:
            print(f"{self.env.now:.2f} - FATAL ERROR: {container} is not in a valid idle state. Adding to generic pool.")
            exit(1)

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

    def update_resource_stats(self):
        """Update time-weighted statistics for resource usage."""
        current_time = self.env.now
        time_delta = current_time - self.last_resource_update
        
        # Calculate current CPU and RAM usage across all servers
        current_cpu_usage = sum(server.cpu_capacity - server.cpu_real  for server in self.servers)
        current_ram_usage = sum(server.ram_capacity - server.ram_real  for server in self.servers)
        
        # Update time-weighted usage areas
        self.total_cpu_usage_area += time_delta * current_cpu_usage
        self.total_ram_usage_area += time_delta * current_ram_usage
        
        self.last_resource_update = current_time
        
    def get_mean_cpu_usage(self):
        """Calculate the mean CPU usage over time."""
        # Update statistics first to include latest data
        self.update_resource_stats()
        
        if self.env.now > 0:
            return self.total_cpu_usage_area / (self.env.now )
        return 0.0
        
    def get_mean_ram_usage(self):
        """Calculate the mean RAM usage over time."""
        # Update statistics first to include latest data
        self.update_resource_stats()
        
        if self.env.now > 0:
            return self.total_ram_usage_area / (self.env.now )
        return 0.0

    def update_processing_stats(self):
        """Update time-weighted statistics for processing requests."""
        current_time = self.env.now
        time_delta = current_time - self.last_processing_update
        self.total_processing_area += time_delta * self.request_running
        self.last_processing_update = current_time
        
    def increment_processing(self):
        """Increment the count of processing requests."""
        self.update_processing_stats()
        self.request_running += 1
        
    def decrement_processing(self):
        """Decrement the count of processing requests."""
        self.update_processing_stats()
        self.request_running -= 1
        
    def get_mean_processing_count(self):
        """Calculate the average number of processing requests over time."""
        # Update statistics first to include latest data
        self.update_processing_stats()
        
        if self.env.now > 0:
            return self.total_processing_area / self.env.now
        return 0.0
        
    def get_mean_requests_in_system(self):
        """Calculate the average number of requests in the system (waiting + processing)."""
        # Update both statistics first to include latest data
        self.update_waiting_stats()
        self.update_processing_stats()
        
        if self.env.now > 0:
            return (self.total_waiting_area + self.total_processing_area) / self.env.now
        return 0.0
        
    def calculate_day_night_arrival_rate(self):
        """Calculate the arrival rate based on a day-night cycle.
        
        Returns a rate that varies based on the time of day:
        - Maximum (peak) at peak_hour (default: noon)
        - Minimum (trough) at midnight
        - Smooth sinusoidal transitions between peak and trough
        """
        # Calculate the time of day (in hours, from 0 to day_length)
        time_of_day = self.env.now % self.day_length
        
        # Calculate the phase of the day (0 to 2π)
        phase = (time_of_day / self.day_length) * 2 * 3.14159
        
        # Calculate the peak hour phase (when traffic is highest)
        peak_phase = (self.peak_hour / self.day_length) * 2 * 3.14159
        
        # Adjust phase so that peak occurs at the defined peak_hour
        adjusted_phase = phase - peak_phase + 3.14159/2
        
        # Calculate scaling factor using a sine wave (varies from 0 to 1)
        # Value of 0 corresponds to trough_factor, value of 1 corresponds to peak_factor
        scaling = (math.sin(adjusted_phase) + 1) / 2
        
        # Interpolate between trough and peak factors
        rate_factor = self.trough_factor + scaling * (self.peak_factor - self.trough_factor)
        
        # Calculate the actual arrival rate
        arrival_rate = self.arrival_rate * rate_factor
        
        if self.verbose and int(self.env.now) % 10 == 0:  # Log every 10 time units
            print(f"{self.env.now:.2f} - Day-night cycle: Time {time_of_day:.1f}h, " 
                  f"Rate factor: {rate_factor:.2f}, Arrival rate: {arrival_rate:.2f}")
            
        return arrival_rate
        
    def calculate_up_down_arrival_rate(self):
        """Calculate the arrival rate based on an up-down pattern.
        
        Traffic pattern where rate linearly increases to a peak during the first half
        of the simulation and then linearly decreases back to the starting rate during 
        the second half.
        
        This pattern requires knowing the total simulation time in advance.
        
        Returns a rate that varies based on the simulation progress:
        - Starts at base_rate * trough_factor
        - Linearly increases to base_rate * peak_factor (at half of simulation time)
        - Linearly decreases back to base_rate * trough_factor (at end of simulation)
        """
        # Get the simulation duration from config, or use a default if not present
        sim_duration = self.config.get("system", {}).get("sim_time", 100.0)
        
        # Calculate the current progress through the simulation (0.0 to 1.0)
        progress = min(self.env.now / sim_duration, 1.0)
        
        # Calculate the rate factor based on simulation progress
        if progress < 0.5:
            # First half: linearly increase from trough_factor to peak_factor
            rate_factor = self.trough_factor + (progress * 2) * (self.peak_factor - self.trough_factor)
        else:
            # Second half: linearly decrease from peak_factor to trough_factor
            rate_factor = self.peak_factor - ((progress - 0.5) * 2) * (self.peak_factor - self.trough_factor)
        
        # Calculate the actual arrival rate
        arrival_rate = self.arrival_rate * rate_factor
        
        if self.verbose and int(self.env.now) % 10 == 0:  # Log every 10 time units
            print(f"{self.env.now:.2f} - Up-down pattern: Progress {progress:.2f}, " 
                  f"Rate factor: {rate_factor:.2f}, Arrival rate: {arrival_rate:.2f}")
            
        return arrival_rate

    def record_request_info(self, request):
        """Record detailed information about a request completion or rejection"""
        request_info = {
            'id': request.id,
            'arrival_time': request.arrival_time,
            'end_time': self.env.now,
            'status': request.state,  # "Finished" for successful requests, "Rejected" for blocked requests
            'waiting_time': request.wait_time,
            'processing_time': (request.end_service_time - request.start_service_time) if hasattr(request, 'end_service_time') else 0,
            'spawn_time': request.spawn_time if hasattr(request, 'spawn_time') else 0
        }
        # print(f"{self.env.now:.2f} - Request {request.id} recorded: {request_info}")
        self.request_records.append(request_info)
        
    def record_resource_snapshot(self):
        """Record a snapshot of system resource usage at the current time"""
        # Only record if at least 1 time unit has passed since the last snapshot
        if self.env.now - self.last_resource_snapshot >= 1.0:
            current_cpu_usage = sum(server.cpu_capacity - server.cpu_real for server in self.servers)
            current_ram_usage = sum(server.ram_capacity - server.ram_real for server in self.servers)
            
            # Calculate usage percentages
            cpu_usage_percent = (current_cpu_usage / self.total_cpu_capacity * 100) if self.total_cpu_capacity > 0 else 0
            ram_usage_percent = (current_ram_usage / self.total_ram_capacity * 100) if self.total_ram_capacity > 0 else 0
            
            resource_snapshot = {
                'time': self.env.now,
                'cpu_usage': current_cpu_usage,
                'ram_usage': current_ram_usage,
                'cpu_usage_percent': cpu_usage_percent,
                'ram_usage_percent': ram_usage_percent,
                'waiting_requests': self.waiting_requests_count,
                'processing_requests': self.request_running
            }
            
            self.resource_history.append(resource_snapshot)
            self.last_resource_snapshot = self.env.now

    def resource_monitor_process(self):
        """Process that periodically records system resource consumption"""
        while True:
            # Record a resource snapshot
            self.record_resource_snapshot()
            
            # Wait for 1 time unit before taking the next snapshot
            yield self.env.timeout(1.0)