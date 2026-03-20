import simpy
import random
import itertools
import csv
import os
import variables
from Request import Request
from Scheduler import FirstFitScheduler  # Import our new scheduler
from LoadBalancer import LoadBalancer  # Import our new LoadBalancer

# --- System Class ---

class System:
    """Orchestrates the simulation, managing servers, requests, and containers."""
    def __init__(self, env, topology, scheduler_class=FirstFitScheduler, trace_path=None):
        self.env = env
        self.topology = topology  # New: topology instance
        self.clusters = topology.clusters  # Dictionary of {cluster_name: cluster_instance}
        self.req_id_counter = itertools.count()
        self.trace_path = trace_path
        # self.verbose = verbose  # Flag to control logging output
        
        # generate idle timeout for applications
        # NOTE: later custom timeout per application per cluster can be implemented
        self.idle_timeout_cluster = {}
        for app_id in variables.APPLICATIONS:
            self.idle_timeout_cluster[app_id] = variables.UNIVERSAL_TIMEOUT
        # Initialize schedulers and idle container pools for each cluster
        self.schedulers = {}
        self.app_idle_containers = {}

        for cluster_name, cluster in self.clusters.items():
            self.schedulers[cluster_name] = scheduler_class(env, cluster, self.idle_timeout_cluster)
            self.app_idle_containers[cluster_name] = {}
            for app_id in variables.APPLICATIONS:
                self.app_idle_containers[cluster_name][app_id] = []
        # Initialize the LoadBalancer (now handling multiple clusters)
        self.load_balancer = LoadBalancer(env, self, self.schedulers)
        
        # For applications containers (now per cluster)
        
        # Create separate idle container pools for each application and cluster
        # for cluster_name in self.clusters:
           

    def request_generator(self):
        """Generates requests for all defined applications."""
        # node_intensity is a percentage (0-100) that determines which level 3 nodes generate requests
        total_request = 0
        for app_id in variables.APPLICATIONS:
            # Calculate log-normal pre-metrics for app spawn time in variables.py
            variables.app_log_normal_metrics(app_id)

            for node_id in self.topology.ingress_nodes:
                # Get node data
                node_data = self.topology.graph.nodes[node_id]
                # Only generate requests with node_intensity probability
                arrival_rate = node_data['population'] * variables.REQ_PER_PERSON
                total_request += arrival_rate
                if arrival_rate > 0:
                    self.env.process(self.app_request_generator(app_id, node_id, arrival_rate))
        print(f"Total expected request arrival rate: {total_request}/time unit.")

    def app_request_generator(self, app_id, node_id, arrival_rate):
        """Generates requests for a specific application according to its Poisson process."""
        # data_location = app_config["data_location"]
        # No defined period, keep generating until simulation time ends
        while True:
            # Time between arrivals (Exponential distribution for Poisson process)
            inter_arrival_time = random.expovariate(arrival_rate)
            yield self.env.timeout(inter_arrival_time)

            # Generate request details
            req_id = next(self.req_id_counter)
            arrival_time = self.env.now
            
            # Generate resource demands for this app
            request_info = variables.generate_req_info(app_id)
            
            # Generate the request
            request = Request(req_id, arrival_time, request_info, node_id, app_id)
            
            # Update statistics
            self.update_start_statistics(request)

            if variables.VERBOSE:
                print(f"{self.env.now:.2f} - Request Generated: {request}")

            # Start the handling process for this request
            self.env.process(self.handle_request(request))

    def handle_request(self, request):
        """Handles an incoming request by delegating to the LoadBalancer."""
        
        # Start measuring waiting time
        request.waiting_start_time = self.env.now
        
        # Find cluster (DC or Edge DC) where request can be processed
        link_found, target_clusters = self.topology.find_cluster(request)

        if not link_found:
            self.update_end_statistics(request, 'link_failed')
            return
        
        # Delegate request handling to the LoadBalancer with viable cluster options
        assignment_result, container, cluster = yield from self.load_balancer.handle_request(request, target_clusters)

        # If assignment was successful, process the service
        if assignment_result:
            # Create the first transmission - upload data to container
            yield from self.topology.update_request_delay(request, \
                                    target_clusters[cluster], type='upload')

            # Start processing the request in the container
            yield from container.process_request()

            # Create the second transmission - download data from container
            yield from self.topology.update_request_delay(request, \
                                    target_clusters[cluster], type='download')

            # Start the idle timeout process 
            container.idle_timeout_process = self.env.process(container.idle_lifecycle())
            
            # Update statistics
            self.update_end_statistics(request, 'success')
        else:
            # print(f"{self.env.now:.2f} - Failed to assign request {request} to a container.")
            self.update_end_statistics(request, 'compute_failed')


    def update_start_statistics(self, request):
        variables.request_stats['generated'] += 1
        # app_stats[request.app_id]['generated'] += 1

    def update_end_statistics(self, request, type):
        if type == 'compute_failed':
            variables.request_stats['blocked_no_server_capacity'] += 1
            # app_stats[request.app_id]['blocked_no_server_capacity'] += 1

        # elif type == 'link_failed':
        #     variables.request_stats['blocked_no_path'] += 1
        #     # app_stats[request.app_id]['blocked_no_path'] += 1
        #     for value in link_failed_map.values():
        #         variables.request_stats['blocked_no_path_level_3-3'] += value.get('3-3', 0)
        #         variables.request_stats['blocked_no_path_level_3-2'] += value.get('3-2', 0)
        #         variables.request_stats['blocked_no_path_level_2-2'] += value.get('2-2', 0)
        #         variables.request_stats['blocked_no_path_level_2-1'] += value.get('2-1', 0)
        #         variables.request_stats['blocked_no_path_level_1-1'] += value.get('1-1', 0)
        #         variables.request_stats['blocked_no_path_level_1-0'] += value.get('1-0', 0)
        #         variables.request_stats['blocked_no_path_level_0-0'] += value.get('0-0', 0)
        else:
            # Service finished
            variables.request_stats['processed'] += 1
            # app_stats[request.app_id]['processed'] += 1
            
            # Record request location
            if request.assigned_cluster == variables.CENTRAL_CLOUD:
                variables.request_stats['offloaded_to_cloud'] += 1
            # Compute latencies: sum of propagation, spawn, and processing times
            total_latency = request.network_delay + request.spawn_time + request.processing_time
            
            # Update global latency stats
            variables.latency_stats['total_latency'] += total_latency
            # variables.latency_stats['propagation_delay'] += request.prop_delay
            variables.latency_stats['spawning_time'] += request.spawn_time
            variables.latency_stats['processing_time'] += request.processing_time
            variables.latency_stats['network_time'] += request.network_delay  # Add network time to stats
            # latency_stats['waiting_time'] += request.waiting_time  # Add waiting time to stats
            variables.latency_stats['count'] += 1

            # Update each request latency
            # Store individual request data for comprehensive analysis
            if variables.SAVE_INDIVIDUAL_LATENCIES:
                variables.accepted_request_latencies.append((
                    request.origin_node,
                    request.arrival_time,
                    request.network_delay,
                    request.spawn_time,
                    total_latency,
                    request.bottleneck
                ))
                
                # Flush to parquet if buffer is full (1 million records)
                if len(variables.accepted_request_latencies) >= 1000000:
                    self.flush_latencies()

            # Update congested path statistics
            if request.bottleneck is not None:
                variables.congested_paths[str(request.bottleneck)] += 1
            # if request.data_path_required and request.bottleneck_indirect is not None:
            #     variables.congested_paths[request.bottleneck_indirect] += 1

            # Update accumulated path latency from request delays
            # for level, delay in request.delay_by_level_direct.items():
            #     accumulated_path_latency[level] += delay/request.network_delay
            
            # if request.data_path_required:
            #     for level, delay in request.delay_by_level_indirect.items():
            #         accumulated_path_latency[level] += delay/request.network_delay
            
            # Update app-specific latency stats
            # app_latency_stats[request.app_id]['total_latency'] += total_latency
            # app_latency_stats[request.app_id]['propagation_delay'] += request.prop_delay
            # app_latency_stats[request.app_id]['spawning_time'] += request.spawn_time
            # app_latency_stats[request.app_id]['processing_time'] += request.processing_time
            # app_latency_stats[request.app_id]['waiting_time'] += request.waiting_time  # Add app-specific waiting time
            # app_latency_stats[request.app_id]['count'] += 1

    def flush_latencies(self):
        """Writes buffered latencies to a CSV file and clears the buffer."""
        columns = ['origin_node', 'arrival_time', 'network_delay', 'spawn_time', 'total_latency', 'bottleneck']
        # Switch trace_path extension to .csv
        csv_path = os.path.splitext(self.trace_path)[0] + '.csv'

        try:
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(columns)
                writer.writerows(variables.accepted_request_latencies)

            print(f"Flushed {len(variables.accepted_request_latencies)} records to {csv_path}")
            variables.accepted_request_latencies.clear()

        except Exception as e:
            print(f"Error flushing latencies to CSV: {e}")


