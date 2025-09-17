import simpy
import random
from variables import VERBOSE

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
              : Strategy for selecting clusters ('edge_first' by default)
            container_selection_strategy: Strategy for selecting containers ('random' by default)
            request_handling_strategy: Strategy for handling requests ('greedy' by default)
            verbose: Flag to control logging output
        """
        self.env = env
        self.system = system
        self.schedulers = schedulers  # Dictionary of {cluster_name: scheduler_instance}
        # self.verbose = verbose  # Flag to control logging output
        
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
        
    def handle_request(self, request, viable_clusters):
        # Handle an incoming request based on the selected strategy.
        
        if self.request_handling_strategy == "greedy":
            return self._handle_request_greedy(request, viable_clusters)
        # Add more strategies here as needed
        else:
            raise ValueError(f"Unknown request handling strategy: {self.request_handling_strategy}")
            
    def _handle_request_greedy(self, request, viable_clusters):
        # Try each viable cluster in order (assuming they are already ordered by propagation delay if topology is used)
        # Get the appropriate container pool for this app in the selected cluster
        for cluster in viable_clusters:
            cluster_name = cluster.name
            idle_container_pool = self.system.app_idle_containers[cluster_name][request.app_id]
            # Try to use an idle container if available
            # idle_container_found = False
            if len(idle_container_pool) > 0:
                container = self.assign_request_to_container(cluster_name, idle_container_pool, request, False)
                if container:
                    return True, container, cluster
            
            # No idle containers available, let the scheduler handle container spawning
            scheduler = self.schedulers[cluster_name]
            # path = paths.get(selected_cluster) if paths else None
            spawn_result, container = yield self.env.process(scheduler.spawn_container_for_request(request, self.system, cluster_name))            
            
            if spawn_result:
                return True, container, cluster
            else:
                if VERBOSE:
                    print(f"{self.env.now:.2f} - No server with sufficient capacity in {cluster_name} cluster for {request}")
                # Continue to next cluster if this one doesn't have resources
        
        # If we've tried all clusters and none worked, the request is blocked
        if VERBOSE:
            print(f"{self.env.now:.2f} - BLOCK: No cluster with sufficient capacity for {request}")
        return False, None, None
                
    # Helper function to process container for request
    def assign_request_to_container(self, cluster_name, idle_container_pool, request, is_new_spawn=False):
        container = idle_container_pool.pop()

        if container.state != "Idle" or container.current_request is not None:
            print(f"FATAL ERROR: {self.env.now:.2f} - Retrieved container {container} is not idle. Discarding it.")
            exit(1)
            
        container.idle_since = -1  # No longer idle
        request.assigned_cluster = cluster_name # save info for request_stats
        # If this container was idle, cancel its removal timeout
        # print(f"{self.env.now:.2f} - {self} status: {self.idle_timeout_process}.")
        if container.idle_timeout_process and not container.idle_timeout_process.triggered:
            if VERBOSE:
                print(f"{self.env.now:.2f} - Cancelling idle timeout for reused {container}")
            container.idle_timeout_process.interrupt()
            container.idle_timeout_process = None  # Clear the process handle

        # msg_type = "Got newly spawned" if is_new_spawn else "Found idle"
        # print(f"{self.env.now:.2f} - {msg_type} {container} in {cluster_name} cluster for {request}")
        # request_stats['containers_reused'] += 1
        # app_stats[request.app_id]['containers_reused'] += 1

        # Simply assign the request without handling resource scaling
        container.current_request = request
        # container.cluster = cluster  # Track which cluster this container belongs to
        return container
