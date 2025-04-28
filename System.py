import simpy
import random
import itertools
from variables import *
from Request import Request
from Container import Container 
from Topology import Topology
from Scheduler import FirstFitScheduler  # Import our new scheduler
from LoadBalancer import LoadBalancer  # Import our new LoadBalancer

# --- System Class ---

class System:
    """Orchestrates the simulation, managing servers, requests, and containers."""
    def __init__(self, env, topology, clusters, use_topology=True, scheduler_class=FirstFitScheduler):
        self.env = env
        self.topology = topology  # New: topology instance
        self.clusters = clusters  # Dictionary of {cluster_name: cluster_instance}
        self.req_id_counter = itertools.count()
        self.use_topology = use_topology  # Flag to control topology usage
        
        # Initialize schedulers for each cluster
        self.schedulers = {}
        for cluster_name, cluster in clusters.items():
            self.schedulers[cluster_name] = scheduler_class(env, cluster)
        
        # Initialize the LoadBalancer (now handling multiple clusters)
        self.load_balancer = LoadBalancer(env, self, self.schedulers)
        
        # For applications containers (now per cluster)
        self.app_idle_containers = {}
        
        # Create separate idle container pools for each application and cluster
        for cluster_name in clusters:
            self.app_idle_containers[cluster_name] = {}
            for app_id in APPLICATIONS:
                self.app_idle_containers[cluster_name][app_id] = simpy.Store(env)

    def request_generator(self):
        """Generates requests for all defined applications."""
        # Start request generators for each application type
        for app_id in APPLICATIONS:
            self.env.process(self.app_request_generator(app_id))

    def app_request_generator(self, app_id):
        """Generates requests for a specific application according to its Poisson process."""
        app_config = APPLICATIONS[app_id]
        arrival_rate = app_config["arrival_rate"]
        
        while True:
            # Time between arrivals (Exponential distribution for Poisson process)
            inter_arrival_time = random.expovariate(arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            # Generate request details
            req_id = next(self.req_id_counter)
            arrival_time = self.env.now
            
            # Generate resource demands for this app
            cpu_warm, ram_warm, cpu_demand, ram_demand = generate_app_demands(app_id)
            
            # For topology-aware simulations, assign a random node from the topology
            origin = None
            if self.use_topology:
                origin = random.choice(list(self.topology.graph.nodes))
                
            request = Request(req_id, arrival_time, cpu_demand, ram_demand, cpu_warm, ram_warm, 
                             origin_node=origin, app_id=app_id)
            
            # Update statistics
            request_stats['generated'] += 1
            app_stats[app_id]['generated'] += 1
            
            print(f"{self.env.now:.2f} - Request Generated: {request}")

            # Start the handling process for this request
            self.env.process(self.handle_request(request))

    def handle_request(self, request):
        """Handles an incoming request by delegating to the LoadBalancer."""
        
        # Start measuring waiting time
        request.waiting_start_time = self.env.now
        
        paths = {}
        target_clusters = []
        
        # If using topology, find viable paths to each cluster
        if self.use_topology:
            # Get app-specific bandwidth demand
            bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
            
            # Find paths to each cluster from the origin node
            for cluster_name, cluster in self.clusters.items():
                path = self.topology.find_path(request.origin_node, cluster.node, bandwidth_demand)
                if path:
                    paths[cluster_name] = path
                    target_clusters.append(cluster_name)
                    
                    # Calculate prop delay for this path (for metrics)
                    prop = sum(self.topology.graph.get_edge_data(path[i], path[i+1])['latency'] 
                                for i in range(len(path)-1))
                    # We'll set this on the request if this cluster is chosen
                    request.potential_prop_delays[cluster_name] = 2 * prop
            
            if not target_clusters:
                print(f"{self.env.now:.2f} - BLOCK: No feasible path from {request.origin_node} to any cluster for {request}")
                request_stats['blocked_no_path'] += 1
                app_stats[request.app_id]['blocked_no_path'] += 1
                return
                
            # Sort target_clusters by propagation delay (lowest first)
            target_clusters.sort(key=lambda cluster_name: request.potential_prop_delays[cluster_name])
            print(f"{self.env.now:.2f} - Ordered clusters by propagation delay for {request}: {target_clusters}")
        else:
            # No topology routing - all clusters are viable targets
            target_clusters = list(self.clusters.keys())
            # Set propagation delay to 0 for all clusters
            for cluster_name in target_clusters:
                request.potential_prop_delays[cluster_name] = 0.0

        # Delegate request handling to the LoadBalancer with viable cluster options
        process = self.load_balancer.handle_request(request, target_clusters, paths)
        result = yield self.env.process(process)
        assignment_result, container, selected_cluster = result
        
        # If assignment was successful, process the service
        if assignment_result:
            # Set the actual propagation delay based on the selected cluster
            request.prop_delay = request.potential_prop_delays[selected_cluster]
            
            # Get the path to the selected cluster
            path = None
            if self.use_topology:
                path = paths.get(selected_cluster, None)
                # Get app-specific bandwidth demand for release
                bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
                self.topology.implement_path(path, bandwidth_demand)

            print(f"{self.env.now:.2f} - The idle state of the container outside is {container.idle_timeout_process}")
            self.env.process(container.service_lifecycle(path, self.topology, self.use_topology))
        else:
            print(f"{self.env.now:.2f} - Failed to assign request {request} to a container.")

    def add_idle_container(self, container, cluster_name):
        """Adds a container to the idle store for a specific cluster."""
        print(f"{self.env.now:.2f} - Adding {container} to idle pool in {cluster_name} cluster.")
        
        # Put container in the appropriate cluster's app idle pool
        self.app_idle_containers[cluster_name][container.app_id].put(container)