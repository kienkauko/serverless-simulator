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
    def __init__(self, env, topology, cluster, use_topology=True, scheduler_class=FirstFitScheduler):
        self.env = env
        self.topology = topology  # New: topology instance
        self.cluster = cluster    # New: cluster instance (holds its servers)
        self.req_id_counter = itertools.count()
        self.use_topology = use_topology  # Flag to control topology usage
        
        # Initialize the scheduler
        self.scheduler = scheduler_class(env, cluster)
        
        # Initialize the LoadBalancer
        self.load_balancer = LoadBalancer(env, self, self.scheduler)
        
        # For applications containers
        self.app_idle_containers = {}
        
        # Create separate idle container pools for each application
        for app_id in APPLICATIONS:
            self.app_idle_containers[app_id] = simpy.Store(env)

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
        
        path = None
        # Handle topology routing if enabled
        if self.use_topology:
            # Get app-specific bandwidth demand
            bandwidth_demand = APPLICATIONS[request.app_id]["bandwidth_demand"]
            
            # First, try to route from request.origin_node to the cluster node.
            path = self.topology.find_path(request.origin_node, self.cluster.node, bandwidth_demand)
            if not path:
                print(f"{self.env.now:.2f} - BLOCK: No feasible path from {request.origin_node} to {self.cluster.node} for {request} (bandwidth demand: {bandwidth_demand})")
                request_stats['blocked_no_path'] += 1
                app_stats[request.app_id]['blocked_no_path'] += 1
                return

            # Compute round-trip propagation delay (sum of latencies *2)
            prop = sum(self.topology.graph.get_edge_data(path[i], path[i+1])['latency'] for i in range(len(path)-1))
            request.prop_delay = 2 * prop
            # print(f"{self.env.now:.2f} - Routed {request} via path: {path} (Propagation delay: {request.prop_delay:.2f})")
        else:
            # No topology routing - set propagation delay to 0
            request.prop_delay = 0.0

        # Delegate request handling to the LoadBalancer
        process = self.load_balancer.handle_request(request, path)
        result = yield self.env.process(process)
        assignment_result, container = result
        # If assignment was successful, process the service
        if assignment_result:
            # Call service lifecycle from container instead of system
            print(f"{self.env.now:.2f} - The idle state of the container outside is {container.idle_timeout_process}")
            self.env.process(container.service_lifecycle(path, self.topology, self.use_topology))
        else:
            print(f"{self.env.now:.2f} - Failed to assign request {request} to a container.")


    def add_idle_container(self, container):
        """Adds a container to the idle store."""
        print(f"{self.env.now:.2f} - Adding {container} to idle pool.")
        
        # Put container in the appropriate app's idle pool
        self.app_idle_containers[container.app_id].put(container)