import simpy
import math
import random
import itertools
from variables import *
from Request import Request
from Scheduler import FirstFitScheduler  # Import our new scheduler
from LoadBalancer import LoadBalancer  # Import our new LoadBalancer

# --- System Class ---

class System:
    """Orchestrates the simulation, managing servers, requests, and containers."""
    def __init__(self, env, topology, clusters, scheduler_class=FirstFitScheduler, verbose=False):
        self.env = env
        self.topology = topology  # New: topology instance
        self.clusters = clusters  # Dictionary of {cluster_name: cluster_instance}
        self.req_id_counter = itertools.count()
        self.verbose = verbose  # Flag to control logging output
        
        # generate idle timeout for applications
        # NOTE: later custom timeout per application per cluster can be implemented
        self.idle_timeout_cluster = {}
        for app_id in APPLICATIONS:
            self.idle_timeout_cluster[app_id] = 2
        # Initialize schedulers for each cluster
        self.schedulers = {}
        for cluster_name, cluster in clusters.items():
            self.schedulers[cluster_name] = scheduler_class(env, cluster, self.idle_timeout_cluster, verbose=self.verbose)
        
        # Initialize the LoadBalancer (now handling multiple clusters)
        self.load_balancer = LoadBalancer(env, self, self.schedulers, verbose=self.verbose)
        
        # For applications containers (now per cluster)
        self.app_idle_containers = {}
        
        # Create separate idle container pools for each application and cluster
        for cluster_name in clusters:
            self.app_idle_containers[cluster_name] = {}
            for app_id in APPLICATIONS:
                self.app_idle_containers[cluster_name][app_id] = []

    def request_generator(self, node_intensity):
        """Generates requests for all defined applications."""
        # node_intensity is a percentage (0-100) that determines which level 3 nodes generate requests
        total_request = 0
        for app_id in APPLICATIONS:
            for node_id, node_data in self.topology.graph.nodes(data=True):
                if node_data['level'] == 3:  # Only level 3 nodes can generate requests
                    # Only generate requests with node_intensity probability
                    if random.random() * 100 < node_intensity:
                        arrival_rate = math.ceil(node_data['population'] * TRAFFIC_INTENSITY)
                        total_request += arrival_rate
                        self.env.process(self.app_request_generator(app_id, node_id, arrival_rate))
        print(f"Total expected request arrival rate: {total_request}.")

    def app_request_generator(self, app_id, node_id, arrival_rate):
        """Generates requests for a specific application according to its Poisson process."""
        app_config = APPLICATIONS[app_id]
        data_location = app_config["data_location"]
        # No defined period, keep generating until simulation time ends
        while True:
            # Time between arrivals (Exponential distribution for Poisson process)
            inter_arrival_time = random.expovariate(arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            # Generate request details
            req_id = next(self.req_id_counter)
            arrival_time = self.env.now
            
            # Generate resource demands for this app
            resource_demand = generate_app_demands(app_id)
            
            # Generate the request
            request = Request(req_id, arrival_time, resource_demand, node_id, data_node = data_location, app_id=app_id)
            
            # Update statistics
            self.update_start_statistics(request)

            if self.verbose:
                print(f"{self.env.now:.2f} - Request Generated: {request}")

            # Start the handling process for this request
            self.env.process(self.handle_request(request))

    def handle_request(self, request):
        """Handles an incoming request by delegating to the LoadBalancer."""
        
        # Start measuring waiting time
        request.waiting_start_time = self.env.now
        
        # paths = {}
        # target_clusters = []
        # Find cluster (DC or Edge DC) where request can be processed
        link_found, target_clusters, failed_links_map = self.topology.find_cluster(request, strategy='always_cloud')

        if not link_found:
            self.update_end_statistics(request, 'link_failed', failed_links_map)
            return
        # Delegate request handling to the LoadBalancer with viable cluster options
        handle_request_process = self.load_balancer.handle_request(request, target_clusters)
        assignment_result, container, cluster = yield self.env.process(handle_request_process)
        
        # If assignment was successful, process the service
        if assignment_result:
            # Start using topology paths
            self.topology.make_paths(request, target_clusters[cluster])
            # Start the service lifecycle 
            yield self.env.process(container.service_lifecycle())
            # Update statistics
            self.update_end_statistics(request, 'success')
            # Release request and resources
            request.state = "Finished"
            container.release_request() # this process also puts container to idle cycle
            self.topology.remove_paths(request, target_clusters[cluster])
            # Put the container into the idle pool
            self.app_idle_containers[cluster_name][container.app_id].append(container)

        else:
            # print(f"{self.env.now:.2f} - Failed to assign request {request} to a container.")
            self.update_end_statistics(request, 'compute_failed')

    # def add_idle_container(self, container, cluster_name):
    #     """Adds a container to the idle store for a specific cluster."""
    #     if self.verbose:
    #         print(f"{self.env.now:.2f} - Adding {container} to idle pool in {cluster_name} cluster.")
    #     # Put container in the appropriate cluster's app idle pool
    #     self.app_idle_containers[cluster_name][container.app_id].append(container)
    
    def update_start_statistics(self, request):
        request_stats['generated'] += 1
        app_stats[request.app_id]['generated'] += 1

    def update_end_statistics(self, request, type, link_failed_map=None):
        if type == 'compute_failed':
            request_stats['blocked_no_server_capacity'] += 1
            app_stats[request.app_id]['blocked_no_server_capacity'] += 1

        elif type == 'link_failed':
            request_stats['blocked_no_path'] += 1
            app_stats[request.app_id]['blocked_no_path'] += 1
            for value in link_failed_map.values():
                request_stats['bocked_no_path_level_3-3'] += value.get('3-3', 0)
                request_stats['bocked_no_path_level_3-2'] += value.get('3-2', 0)
                request_stats['bocked_no_path_level_2-2'] += value.get('2-2', 0)
                request_stats['bocked_no_path_level_2-1'] += value.get('2-1', 0)
                request_stats['bocked_no_path_level_1-1'] += value.get('1-1', 0)
                request_stats['bocked_no_path_level_1-0'] += value.get('1-0', 0)
                request_stats['bocked_no_path_level_0-0'] += value.get('0-0', 0)
        else:
            # Service finished
            request_stats['processed'] += 1
            app_stats[request.app_id]['processed'] += 1
            
            # Compute latencies: sum of propagation, spawn, and processing times
            total_latency = request.prop_delay + request.spawn_time + request.processing_time
            
            # Update global latency stats
            latency_stats['total_latency'] += total_latency
            latency_stats['propagation_delay'] += request.prop_delay
            latency_stats['spawning_time'] += request.spawn_time
            latency_stats['processing_time'] += request.processing_time
            latency_stats['waiting_time'] += request.waiting_time  # Add waiting time to stats
            latency_stats['count'] += 1
            
            # Update app-specific latency stats
            app_latency_stats[request.app_id]['total_latency'] += total_latency
            app_latency_stats[request.app_id]['propagation_delay'] += request.prop_delay
            app_latency_stats[request.app_id]['spawning_time'] += request.spawn_time
            app_latency_stats[request.app_id]['processing_time'] += request.processing_time
            app_latency_stats[request.app_id]['waiting_time'] += request.waiting_time  # Add app-specific waiting time
            app_latency_stats[request.app_id]['count'] += 1
        
        
       