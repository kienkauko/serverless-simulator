import simpy
import random
from variables import *

class LoadBalancer:
    """Handles assignment of requests to containers using different strategies."""
    
    def __init__(self, env, system, schedulers, 
                 cluster_selection_strategy="edge_first",
                 container_selection_strategy="random",
                 request_handling_strategy="greedy"):
        """Initialize the load balancer with configurable strategies.
        
        Args:
            env: SimPy environment
            system: System instance
            schedulers: Dictionary of {cluster_name: scheduler_instance}
            cluster_selection_strategy: Strategy for selecting clusters ('edge_first' by default)
            container_selection_strategy: Strategy for selecting containers ('random' by default)
            request_handling_strategy: Strategy for handling requests ('greedy' by default)
        """
        self.env = env
        self.system = system
        self.schedulers = schedulers  # Dictionary of {cluster_name: scheduler_instance}
        
        # Set strategies
        self.cluster_selection_strategy = cluster_selection_strategy
        self.container_selection_strategy = container_selection_strategy
        self.request_handling_strategy = request_handling_strategy
        
    def select_cluster(self, request, viable_clusters, paths=None):
        """
        Select which cluster should handle a request based on the selected strategy.
        
        Args:
            request: The request to be handled
            viable_clusters: List of cluster names that can handle the request
            paths: Dictionary of paths to each cluster (for topology awareness)
            
        Returns:
            string: Name of the selected cluster
        """
        if self.cluster_selection_strategy == "edge_first":
            return self._edge_first_selection(request, viable_clusters, paths)
        # Add more strategies here as needed
        else:
            raise ValueError(f"Unknown cluster selection strategy: {self.cluster_selection_strategy}")
            
    def _edge_first_selection(self, request, viable_clusters, paths=None):
        """Edge-first cluster selection strategy.
        
        Prefers edge cluster if available, then cloud, then random selection.
        """
        if 'edge' in viable_clusters:
            # Prefer edge for latency-sensitive applications
            return 'edge'
        elif 'cloud' in viable_clusters:
            # Fall back to cloud if edge isn't available
            return 'cloud'
        else:
            # If neither edge nor cloud specifically, choose randomly
            return random.choice(viable_clusters)
    
    def get_container_for_request(self, idle_container_pool):
        """
        Get a container for a request from the idle pool based on selected strategy.
        
        Args:
            idle_container_pool: Pool of idle containers
            
        Returns:
            Container or None if no container is available
        """
        if self.container_selection_strategy == "random":
            return self._get_random_container(idle_container_pool)
        # Add more strategies here as needed
        else:
            raise ValueError(f"Unknown container selection strategy: {self.container_selection_strategy}")
    
    def _get_random_container(self, idle_container_pool):
        """Random container selection strategy.
        
        Selects a random container from the idle pool if available.
        """
        if len(idle_container_pool.items) > 0:
            # Get all containers and choose randomly
            containers = list(idle_container_pool.items)
            return random.choice(containers)
        return None
        
    def handle_request(self, request, viable_clusters, paths=None):
        """
        Handle an incoming request based on the selected strategy.
        
        Args:
            request: The request to be handled
            viable_clusters: List of cluster names that can handle the request
            paths: Dictionary of {cluster_name: path} for topology routing
            
        Returns:
            tuple: (success_flag, container, selected_cluster)
        """
        if self.request_handling_strategy == "greedy":
            return self._handle_request_greedy(request, viable_clusters, paths)
        # Add more strategies here as needed
        else:
            raise ValueError(f"Unknown request handling strategy: {self.request_handling_strategy}")
            
    def _handle_request_greedy(self, request, viable_clusters, paths=None):
        """
        Greedy strategy for handling requests: try each viable cluster in order.
        
        Args:
            request: The request to be handled
            viable_clusters: List of cluster names that can handle the request, ordered by priority
            paths: Dictionary of {cluster_name: path} for topology routing
            
        Returns:
            tuple: (success_flag, container, selected_cluster)
        """
        # Try each viable cluster in order (assuming they are already ordered by propagation delay if topology is used)
        for selected_cluster in viable_clusters:
            request.assigned_cluster = selected_cluster
            
            print(f"{self.env.now:.2f} - Trying {selected_cluster} cluster for {request}")
            path = paths.get(selected_cluster) if paths else None

            # Get the appropriate container pool for this app in the selected cluster
            idle_container_pool = self.system.app_idle_containers[selected_cluster][request.app_id]
            
            # Try to use an idle container if available
            # idle_container_found = False
            while len(idle_container_pool.items) > 0:
                container = yield self.env.process(self.assign_request_to_container(idle_container_pool, request, selected_cluster))
                if container:
                    if self.system.use_topology and path:
                    # Get app-specific bandwidth demand for release
                        bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
                        self.system.topology.implement_path(path, bandwidth_demand)
                    return True, container, selected_cluster
            
            # No idle containers available, let the scheduler handle container spawning
            scheduler = self.schedulers[selected_cluster]
            # path = paths.get(selected_cluster) if paths else None
            spawn_result = scheduler.spawn_container_for_request(request, self.system, selected_cluster)
            
            if spawn_result:
                # Wait for a container to appear in the idle pool (this will be our newly spawned container)
                while True:
                    container = yield self.env.process(self.assign_request_to_container(idle_container_pool, request, selected_cluster, True))
                    print(f"{self.env.now:.2f} - container: {container}")
                    if container:
                        print(f"{self.env.now:.2f} - The idle state of the container is {container.idle_timeout_process}")
                        if self.system.use_topology and path:
                    # Get app-specific bandwidth demand for release
                            bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
                            self.system.topology.implement_path(path, bandwidth_demand)
                        return True, container, selected_cluster
            else:
                print(f"{self.env.now:.2f} - BLOCK: No server with sufficient capacity in {selected_cluster} cluster for {request}")
                # Continue to next cluster if this one doesn't have resources
                continue
        
        # If we've tried all clusters and none worked, the request is blocked
        print(f"{self.env.now:.2f} - BLOCK: No cluster with sufficient capacity for {request}")
        request_stats['blocked_no_server_capacity'] += 1
        app_stats[request.app_id]['blocked_no_server_capacity'] += 1
        
        return False, None, request.assigned_cluster
                
    # Helper function to process container for request
    def assign_request_to_container(self, idle_container_pool, request, cluster_name, is_new_spawn=False):
        get_op = idle_container_pool.get()
        container = yield get_op

        if container.state != "Idle":
            print(f"WARNING: {self.env.now:.2f} - Retrieved container {container} is not idle. Discarding it.")
            return None
            
        # Make sure container matches the request's app
        # if container.app_id != request.app_id or container.current_request:
        #     print(f"FATAL ERROR: {self.env.now:.2f} - Container {container} app does not match {request}. Discarding it.")
        #     exit(1)

        # Add a small timeout to make this a generator function
        print(f"{self.env.now:.2f} - Start assigning request {request} to container {container}.")
        assign_time = random.expovariate(CONTAINER_ASSIGN_RATE)
        yield self.env.timeout(assign_time)
        
        # Final verification that container is still usable after assignment time
        if container.state == "Dead":
            print(f"{self.env.now:.2f} - WARNING: Container {container} died during assignment process. Cannot use it.")
            return None

        container.idle_since = -1  # No longer idle
        
        # If this container was idle, cancel its removal timeout
        # print(f"{self.env.now:.2f} - {self} status: {self.idle_timeout_process}.")
        if container.idle_timeout_process and not container.idle_timeout_process.triggered:
            print(f"{self.env.now:.2f} - Cancelling idle timeout for reused {container}")
            container.idle_timeout_process.interrupt()
            container.idle_timeout_process = None  # Clear the process handle

        msg_type = "Got newly spawned" if is_new_spawn else "Found idle"
        print(f"{self.env.now:.2f} - {msg_type} {container} in {cluster_name} cluster for {request}")
        request_stats['containers_reused'] += 1
        app_stats[request.app_id]['containers_reused'] += 1

        # Simply assign the request without handling resource scaling
        container.current_request = request
        container.cluster_name = cluster_name  # Track which cluster this container belongs to
        return container
