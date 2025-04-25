import simpy
import random
from variables import *

class LoadBalancer:
    """Handles assignment of requests to containers using different strategies."""
    
    def __init__(self, env, system, scheduler):
        """Initialize the load balancer."""
        self.env = env
        self.system = system
        self.scheduler = scheduler
        
    def get_container_for_request(self, idle_container_pool):
        """Get a container for a request from the idle pool.
        
        Currently implements a simple random selection strategy.
        Other strategies can be implemented later.
        """
        # Default strategy: random selection of same-type container
        if len(idle_container_pool.items) > 0:
            # Get all containers and choose randomly
            containers = list(idle_container_pool.items)
            return random.choice(containers)
        return None
        
    def handle_request(self, request, path=None):
        """Handle an incoming request: find idle container or spawn a new one."""
        """Current implementation: if no idle container, spawn a new one. But
        request may not use the new container if there exists an idle one in the pool."""
        # Get the appropriate container pool for this app
        idle_container_pool = self.system.app_idle_containers[request.app_id]
        
        # Try to use an idle container if available
        while len(idle_container_pool.items) > 0:
            container = yield self.env.process(self.assign_request_to_container(idle_container_pool, request))
            if container:
                return True, container
            
        # No idle containers available, let the scheduler handle container spawning
        spawn_result = self.scheduler.spawn_container_for_request(request, self.system, path)
        
        if spawn_result:
            # Wait for a container to appear in the idle pool (this will be our newly spawned container)
            while True:
                container = yield self.env.process(self.assign_request_to_container(idle_container_pool, request, True))
                print(f"{self.env.now:.2f} - container: {container}")
                if container:
                    print(f"{self.env.now:.2f} - The idle state of the container is {container.idle_timeout_process}")
                    return True, container
        else:
            print(f"{self.env.now:.2f} - BLOCK: No server with sufficient capacity for {request}")
            request_stats['blocked_no_server_capacity'] += 1
            app_stats[request.app_id]['blocked_no_server_capacity'] += 1
                
            if self.system.use_topology and path:
                # Get app-specific bandwidth demand for release
                bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
                self.system.topology.release_path(path, bandwidth_demand)
            return False, None  # Return failure status and no container
                
    # def assign_request_to_container(self, container, request):
    #     """Assigns a request to a container. Returns a generator for SimPy process."""
        
    # Helper function to process container for request
    def assign_request_to_container(self, idle_container_pool, request, is_new_spawn=False):
        get_op = idle_container_pool.get()
        container = yield get_op

        if container.state != "Idle":
            print(f"WARNING: {self.env.now:.2f} - Retrieved container {container} is not idle. Discarding it.")
            return False
            
        # Make sure container matches the request's app
        if container.app_id != request.app_id or container.current_request:
            print(f"FATAL ERROR: {self.env.now:.2f} - Container {container} app does not match {request}. Discarding it.")
            exit(1)

        msg_type = "Got newly spawned" if is_new_spawn else "Found idle"
        print(f"{self.env.now:.2f} - {msg_type} {container} for {request}")
        request_stats['containers_reused'] += 1
        app_stats[request.app_id]['containers_reused'] += 1

        # Add a small timeout to make this a generator function
        print(f"{self.env.now:.2f} - Start assigning request {request} to container {container}.")
        assign_time = random.expovariate(CONTAINER_ASSIGN_RATE)
        yield self.env.timeout(assign_time)
        
        # Simply assign the request without handling resource scaling
        container.current_request = request
        return container
