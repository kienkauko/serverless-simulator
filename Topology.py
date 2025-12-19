import networkx as nx
import json
import math
import random
from pyproj import Transformer
from Cluster import Cluster 
from FlowState import FlowState
import variables
import simpy

class Topology:
    def __init__(self, env):
        # self.clusters = clusters
        # Store edge cluster for easy access if provided
        # self.edge_cluster = [cluster for cluster_name, cluster in clusters.items() if cluster_name == "edge"] if clusters else []
        self.bw_factor_enable = variables.BW_POPULATION_SCALE_ENABLE
        self.bandwidth_factor = variables.REQ_PER_PERSON # Scale following the traffic intensity
        self.link_util_enable = variables.LINK_UTILIZATION_ENABLE
        self.link_utilization = variables.LINK_UTILIZATION
        print("Country: ", variables.COUNTRY_CODE)
        print(f"Initializing topology..."
              f"bandwidth factor enabled: {self.bw_factor_enable}, "
              f"link utilization enabled: {self.link_util_enable}")
        # Initialize Transformer for later propagation delay calculations
        self.transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Initialize graph
        self.graph = nx.DiGraph()
        self.env = env

        # Define Ingress nodes
        self.ingress_nodes = []

        # Add a path cache
        self.path_cache = {}  # Format: (src, dst) -> path
        self.prop_delay_cache = {}  # Format: (tuple(path)) -> prop delay

        # Flow registry: track active transmission processes per link
        # Format: {(node1, node2): [list of FlowState objects]}
        self.active_flows = {}

        # Bandwidth registry: store bandwidth for each link
        # Format: {(node1, node2): bandwidth}
        self.link_bw_capacity = {}

        # Current allocated bandwidth per flow on each link
        self.link_allocated_bw = {}

        # Link level registry: store level for each link
        # Format: {(node1, node2): level}
        self.link_level = {}

        # NOTE: the following variables are used for periodic-interruption mechanism
        # used in update_request_delay to reduce runtime
        self.interruption_period = variables.NETWORK_UPDATE_PERIOD  # Time units between interruptions
        self.last_interruption_time = 0  # Last time an interruption occurred

        # Initialize clusters
        self.clusters = {}
        self.cloud_clusters = {}
        self.edge_clusters = {}

        # Add DC mapping structure, use only for per_ring strategy!
        self.node_dc_mapping = {}  # Maps node_id -> [list of DC node_ids]
        
        # variables only to print out
        self.population = 0
        # Load edge data from JSON (includes both node and link data)
        with open(variables.TOPOLOGY_PATH, 'r') as f:
            edge_data = json.load(f)
        
        # Process nodes from edge.json
        for node in edge_data.get('nodes', []):
            node_name = node['name']
            node_id = node_name.replace('_R0', '')  # Remove _R0 suffix
            node_population = node['population']
            if variables.POPULATION_SCALE_ENABLE:
                node_population = node_population * (variables.DEU_POPU / variables.POPULATION_MAP[variables.COUNTRY_CODE])
                if node['level'] == 3:
                    self.population += node_population
            # Add node to the graph with all attributes
            self.graph.add_node(node_id, 
                                location=node['location'],
                                level=node['level'],
                                population=node_population,
                                parent=str(node.get('parent', None)),
                                nearby_clusters=None)  # Will be updated later if needed
        
        # Print for debugging
        if variables.POPULATION_SCALE_ENABLE:
            print(f"Total scaled population after scaling: {self.population}")
        # Process links from edge.json
        for link in edge_data.get('links', []):
            n1_name = link['n1']
            n2_name = link['n2']
            
            # Extract node IDs
            n1_id = n1_name.replace('_R0', '')
            n2_id = n2_name.replace('_R0', '')
            
            # Skip self-loops
            if n1_id == n2_id:
                continue
            # if n1_id == "6718" and n2_id == "12817":
            #     pass
            # Check if both nodes exist in our graph
            if n1_id in self.graph.nodes and n2_id in self.graph.nodes:
                # Get link level
                node1_level = self.graph.nodes[n1_id]['level']
                node2_level = self.graph.nodes[n2_id]['level']
                combined_level = f"{node1_level}-{node2_level}"

                # Calculate latency based on node locations
                latency = self.calculate_prop_delay(n1_id, n2_id)
                
                # Get bandwidth from JSON or use default
                # Current bandwidth is scaled by REQ_PER_PERSON
                if self.bw_factor_enable:
                    bandwidth = float(link.get('bandwidth'))*self.bandwidth_factor 
                else:
                    bandwidth = float(link.get('bandwidth'))
                # Apply link utilization if enabled
                if self.link_util_enable:
                    utilization = self.link_utilization[combined_level]
                    bandwidth = bandwidth * (1 - utilization)
                
                # Add edge to the graph with attributes 
                self.graph.add_edge(n1_id, n2_id, 
                                       bandwidth=bandwidth, 
                                       latency=latency,
                                       level=combined_level)
                
                # Store bandwidth and level in the separate dictionaries
                self.link_bw_capacity[(n1_id, n2_id)] = bandwidth
                self.link_level[(n1_id, n2_id)] = combined_level

        # Load predefined clusters information from JSON
        for cluster in edge_data['clusters']:
            cluster_name = cluster['name']
            self.clusters[cluster_name] = Cluster(self.env, cluster)
            if "cloud" in cluster_name:
                self.cloud_clusters[cluster_name] = self.clusters[cluster_name]
            elif "edge" in cluster_name:
                self.edge_clusters[cluster_name] = self.clusters[cluster_name]
            else:
                print(f"Warning: Unrecognized cluster type for {cluster_name}")
            

        # Initialize edge DCs if strategy requires it
        if "edge" in variables.CLUSTER_STRATEGY:
            print("Edge strategy is: ", variables.CLUSTER_STRATEGY)
            print("Edge DC level is: ", variables.EDGE_DC_LEVEL)
            self.edge_clusters = self.defined_edge_DCs(variables.EDGE_DC_LEVEL,  variables.CLUSTER_STRATEGY, \
                                                        variables.EDGE_SERVER_PROVISION_STRATEGY, \
                                                        variables.NUM_DC_PER_RING)
            self.clusters.update(self.edge_clusters)
            # As we assume requests come from layer-3 node, we need to define which clusters can
            # these requests reach, they are called nearby clusters
            for node_id in [n for n, data in self.graph.nodes(data=True) if data.get('level') == 3]:
                nearby_clusters = self.get_nearby(variables.CLUSTER_STRATEGY, node_id, variables.EDGE_DC_LEVEL)
                self.graph.nodes[node_id]['nearby_clusters'] = nearby_clusters

        # Define ingress nodes
        self.define_ingress_nodes(mode=variables.INGRESS_MODE)

    
    def define_ingress_nodes(self, mode = 'country'):
        """Define ingress nodes based on the specified mode."""
        if mode == 'country':
            # Define ingress nodes as all level 3 nodes
            # print("Defined ingress nodes for: country.")
            self.ingress_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('level') == 3]
        elif mode == 'cities':
            # Get list of cities (node 1) from variables
            node_1_list = list(variables.EXAMINED_CITIES.values())
            city_names = list(variables.EXAMINED_CITIES.keys())
             # Find all level 3 nodes whose grandparents (layer 1) are in node_1_list
            for node, data in self.graph.nodes(data=True):
                if data.get('level') == 3:
                    # Get grandparent (layer 1 node)
                    grandparent = self.graph.nodes[data.get('parent')].get('parent')
                    # Check if grandparent is in the examined cities list
                    if grandparent in node_1_list:
                        self.ingress_nodes.append(node)
            # print(f"Defined ingress nodes for: {city_names}")
        print(f"Total ingress nodes defined: {len(self.ingress_nodes)}")



        
    # def get_link_utilization(self):
    #     """Calculate and return average link utilization statistics grouped by level."""
    #     utilization_stats = {}
    #     level_edges = {}
        
    #     # Collect utilization data for each edge grouped by level
    #     for n1, n2, data in self.graph.edges(data=True):
    #         bandwidth = data.get('bandwidth', 0)
    #         available_bw = data.get('available_bandwidth', 0)
    #         level = data.get('level', 'unknown')
            
    #         # Calculate utilization as percentage
    #         utilization = ((bandwidth - available_bw) / bandwidth) * 100 
            
    #         if level not in level_edges:
    #             level_edges[level] = {
    #                 'count': 0,
    #                 'total_utilization': 0
    #             }
            
    #         level_edges[level]['count'] += 1
    #         level_edges[level]['total_utilization'] += utilization
        
    #     # Calculate average utilization for each level
    #     for level, stats in level_edges.items():
    #         if stats['count'] > 0:
    #             utilization_stats[level] = stats['total_utilization'] / stats['count']
        
    #     return utilization_stats
    
    def calculate_prop_delay(self, node1, node2):
        """Calculate latency between two nodes based on their locations in the JSON file."""
        # Get locations in EPSG:3857
        loc1 = self.graph.nodes[node1]['location']
        loc2 = self.graph.nodes[node2]['location']
        
        # OPTIMIZATION: Use pre-initialized transformer
        lon1, lat1 = self.transformer.transform(loc1[0], loc1[1])
        lon2, lat2 = self.transformer.transform(loc2[0], loc2[1])
        
        # Calculate distance using Haversine formula (great-circle distance)
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Difference in coordinates
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        
        # Haversine formula
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance_km = R * c
        
        # Calculate latency: Assume speed of light in fiber is ~200,000 km/s
        # This gives us 0.005 ms per km (1000/200000)
        latency_ms = distance_km * 0.005
        
        return latency_ms/1000  # Convert to seconds

    
    def find_possible_path(self, src, dst):
        """Find shortest path between two nodes considering bandwidth if specified."""
        path = self.get_cached_path(src, dst)
        if not path:
            path = self.find_hierachical_path(src, dst)
            
        if path:
            self.save_cached_path(src, dst, path)
            return True, path
        
        else:
            return False, None
            
    
    def find_hierachical_path(self, src, dst):
        # Case: src and dst are at different levels
        src_level = self.graph.nodes[src]['level']
        dst_level = self.graph.nodes[dst]['level']
        
        if src_level != dst_level:
            higher_level = max(src_level, dst_level)
            lower_level = min(src_level, dst_level)

            node_a = src if self.graph.nodes[src]['level'] == higher_level else dst
            node_b = dst if node_a == src else src

            node_a_parent = self.graph.nodes[src]['parent']

            path = [node_a]
            
            while self.graph.nodes[node_a]['level'] > lower_level:
                if self.graph.nodes[node_a]['level'] == 1:
                    path += [node_a_parent]
                    if node_a_parent != node_b:
                        path += [node_b]
                    return path
                cached_path = self.get_cached_path(node_a, node_a_parent)
                if cached_path:
                    path += cached_path[1:]  # Avoid duplicating node_a
                else:
                    try:
                        cached_path = nx.shortest_path(self.graph, node_a, node_a_parent, weight=None)
                        path += cached_path[1:]
                        self.save_cached_path(node_a, node_a_parent, cached_path)
                    except nx.NetworkXNoPath:
                        return None
                node_a = node_a_parent
                node_a_parent = self.graph.nodes[node_a].get('parent', None)
            return path
                
        # Case routing from same and 0 level
        elif src_level == dst_level and src_level == 0:
            if src == dst:
                return [src]
            else:
                return [src, dst]
        # Case routing from same and high levels
        else:
            # Check parent first
            src_parent = self.graph.nodes[src]['parent']
            dst_parent = self.graph.nodes[dst]['parent']
            # If src.parent == dst.parent then do simple_paths
            if src_parent == dst_parent:
                return nx.shortest_path(self.graph, src, dst, weight='latency')
            else:
                # Find paths to each parent first, then check parents again
                path_1 = nx.shortest_path(self.graph, src, src_parent, weight='latency')
                # paths.append(path)
                path_2 = nx.shortest_path(self.graph, dst_parent, dst, weight='latency')
                # paths.append(path)
                # Check if parents are connected through grandparent
                src_grandparent = self.graph.nodes[src_parent]['parent']
                dst_grandparent = self.graph.nodes[dst_parent]['parent']
                if src_grandparent == dst_grandparent:
                    path_3 = nx.shortest_path(self.graph, src_parent, dst_parent, weight='latency')
                    path_final = path_1 + path_3[1:] + path_2[1:]
                    return path_final
                else:
                    # Find paths to each grandparent first
                    path_4 = nx.shortest_path(self.graph, src_parent, src_grandparent, weight='latency')
                    # paths.append(path)
                    path_5 = nx.shortest_path(self.graph, dst_grandparent, dst_parent, weight='latency')
                    # paths.append(path)
                    # get info about great-grandparents
                    src_ggparent = self.graph.nodes[src_grandparent]['parent']
                    dst_ggparent = self.graph.nodes[dst_grandparent]['parent']
                    # find paths to great-grandparents
                    if self.graph.has_edge(src_grandparent, src_ggparent) and \
                        self.graph.has_edge(dst_grandparent, dst_ggparent):
                        # check if grandparents are connected through great-grandparent
                        if src_ggparent == dst_ggparent:
                            path_final = path_1 + path_4[1:] + [src_ggparent] + path_5 + path_2[1:]
                            return path_final
                        # paths.append([src_grandparent, src_ggparent], [dst_ggparent, dst_grandparent])
                        else:
                            # simply add last path connecting ggparent together
                            if self.graph.has_edge(src_ggparent, dst_ggparent):
                                path_final = path_1 + path_4[1:] + [src_ggparent, dst_ggparent] + path_5 + path_2[1:]
                                return path_final
                            else:
                                print(f"Error in finding path: greate grandparents")
                                exit(1)
                    else:
                        print(f"Error in finding path")
                        exit(1)
         
    

    def calculate_TCP_overhead(self, prop_delay, packet_size):
        """
        Calculate TCP overhead including connection setup, teardown, and slow start delay.
        """
        # Calculate RTT in seconds
        rtt = 2 * prop_delay
        
        # Connection setup (3-way handshake) + teardown (simplified to 1 RTT)
        tcp_connection_overhead = 2 * rtt
        
        # Calculate slow start delay
        IW_bits = 10 * 1460 * 8  # Initial Window: 10 MSS, 1 MSS = 1460 bytes = 11680 bits
        num_rtts_slow_start = math.ceil(math.log2(packet_size / IW_bits + 1))
        slow_start_delay = num_rtts_slow_start * rtt
        
        return tcp_connection_overhead, slow_start_delay

    def update_request_delay(self, request, paths, type='upload'):
        """
        Calculate propagation and transmission delays with dynamic bandwidth reallocation.
        This is an interruptible SimPy process that recalculates delay when other flows start/end.
        """
        if type == 'upload':
            path_direct, path_indirect = paths[0], paths[1]
            packet_size_direct = request.packet_size_direct_upload
            packet_size_indirect = request.packet_size_indirect_upload
        elif type == 'download':
            path_direct, path_indirect = paths[0][::-1], paths[1][::-1]
            packet_size_direct = request.packet_size_direct_download
            packet_size_indirect = request.packet_size_indirect_download
        else:
            raise ValueError(f"Unknown delay type: {type}")
        
        # Early return if path is too short
        if not path_direct or len(path_direct) < 2:
            # Only calculate TCP overhead, no propagation or transmission delay
            tcp_connection_overhead, slow_start_delay = self.calculate_TCP_overhead(0, packet_size_direct)
            total_delay = tcp_connection_overhead + slow_start_delay
            request.network_delay += total_delay
            return
        
        # Static propagation delay (one-time calculation)
        prop_delay = self.get_path_prop_delay(path_direct)
        if request.data_path_required:
            prop_delay += self.get_path_prop_delay(path_indirect)
        
        # Calculate TCP overhead (one-time calculation)
        tcp_connection_overhead, slow_start_delay = self.calculate_TCP_overhead(prop_delay, packet_size_direct)
        
        pre_transmission_delay = prop_delay + tcp_connection_overhead + slow_start_delay
        yield self.env.timeout(pre_transmission_delay)

        # Account for propagation and TCP setup delays
        request.prop_delay += prop_delay  
        
        # Get reference to the current process
        current_process = self.env.active_process
        
        # Initialize flow states for transmission
        flow_direct = FlowState(request, path_direct, packet_size_direct, 'direct', current_process)
        if request.data_path_required:
            flow_indirect = FlowState(request, path_indirect, packet_size_indirect, 'indirect', current_process)
        
        # Start transmission timing after propagation and TCP setup
        transmission_start = self.env.now
        flow_direct.start_time = transmission_start
        flow_direct.last_update_time = transmission_start

        # Register flows with the network
        self._register_flow(flow_direct)

         # Get first time allocated bandwidth and bottleneck info
        # self._update_allocated_bandwidth(flow_direct)

        # Interrupt other flows on shared links

        # NOTE: The following interruption per period is done to reduce the runtime, it
        # decrease the accuracy inproportinal to the set period.
        # PLEASE USE WITH CAUTION!
        if transmission_start - self.last_interruption_time >= self.interruption_period:
            self._interrupt_affected_flows(current_process, path_direct)
            self.last_interruption_time = transmission_start

        if request.data_path_required:
            flow_indirect.start_time = transmission_start
            flow_indirect.last_update_time = transmission_start
            self._register_flow(flow_indirect)
            self._interrupt_affected_flows(current_process, path_indirect)

        # Transmission loop - continues until all data is transmitted
        while flow_direct.data_remaining > 0 or (flow_indirect and flow_indirect.data_remaining > 0):
            try:
                # Update allocated bandwidth
                self._update_allocated_bandwidth(flow_direct)

                # Calculate expected delay based on current network state
                trans_delay_direct =  flow_direct.data_remaining / flow_direct.min_bandwidth
                # bottleneck_direct = flow_direct.bottleneck_level
                
                if request.data_path_required:
                    trans_delay_indirect = flow_indirect.data_remaining / flow_indirect.min_bandwidth
                    # bottleneck_indirect = flow_indirect.bottleneck_level
                else:
                    trans_delay_indirect = 0

                # The transmission time is determined by the slower flow (bottleneck)
                expected_delay = max(trans_delay_direct, trans_delay_indirect)
                
                # NOTE: Store bottleneck info, assume that we only care about the first bottleneck encountered
                # max_trans_delay = expected_delay
                # if max_trans_delay > request.max_trans_delay:
                    # request.max_trans_delay = max_trans_delay
                # if request.bottleneck is None:
                #     request.bottleneck = bottleneck_direct if expected_delay == trans_delay_direct else bottleneck_indirect
                
                # Wait for the expected transmission time
                yield self.env.timeout(expected_delay)
                
                # If we reach here without interruption, transmission is complete
                flow_direct.data_remaining = 0
                break
                
            except simpy.Interrupt:
                # Another flow started/ended, affecting our bandwidth allocation
                # Update progress for both flows based on time elapsed
                current_time = self.env.now
                
                # Update direct flow progress
                # if flow_direct.get_remaining_data() > 0:
                # allocated_bw, _ = self._get_allocated_bandwidth(flow_direct)
                flow_direct.update_progress(current_time)
                
                # Update indirect flow progress
                if request.data_path_required:
                    # allocated_bw, _ = self._get_allocated_bandwidth(flow_indirect)
                    flow_indirect.update_progress(current_time)
                
                # Loop continues to recalculate delay with new bandwidth allocation
        
        # Unregister flows from the network
        self._unregister_flow(flow_direct)        
        # Interrupt other flows that were sharing our links (their bandwidth just increased)
         # NOTE: The following interruption per period is done to reduce the runtime, it
        # decrease the accuracy inproportinal to the set period.
        # PLEASE USE WITH CAUTION!
        end_time = self.env.now
        if end_time - self.last_interruption_time >= self.interruption_period:
            self._interrupt_affected_flows(current_process, path_direct)
            self.last_interruption_time = end_time

        if request.data_path_required:
            self._unregister_flow(flow_indirect)
            self._interrupt_affected_flows(current_process, path_indirect)
        
        # Calculate total delay including propagation, TCP overhead, and transmission
        total_delay = pre_transmission_delay + (self.env.now - transmission_start)
        request.network_delay += total_delay
        
        # NOTE: We assume to save bottleneck of the first connection only, result returning
        # connection's bottleneck isn't recorded (assummingly it's very small)
        if request.bottleneck is None:
            request.bottleneck = flow_direct.bottleneck_level

    def _register_flow(self, flow_state):
        """Register a flow with all links in its path."""
        path = flow_state.path
        
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            if link not in self.active_flows:
                self.active_flows[link] = set() # OPTIMIZATION: Use set instead of list
            self.active_flows[link].add(flow_state) # OPTIMIZATION: Use add instead of append

            # Update cached bandwidth for this link
            self.link_allocated_bw[link] = self.link_bw_capacity[link] / len(self.active_flows[link])

    def _unregister_flow(self, flow_state):
        """Unregister a flow from all links in its path."""
        path = flow_state.path
        
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            # OPTIMIZATION: Removal is now O(1) instead of O(N)
            if link in self.active_flows and flow_state in self.active_flows[link]:
                self.active_flows[link].remove(flow_state)

            # Update cached bandwidth for this link
            count = len(self.active_flows[link])
            if count > 0:
                self.link_allocated_bw[link] = self.link_bw_capacity[link] / count
            else:
                self.link_allocated_bw[link] = self.link_bw_capacity[link]
                
    def _interrupt_affected_flows(self, current_process, path):
        """Interrupt only selected flows that share links with the given path and got it min_bandwidth changed."""
        # path = current_flow.path
        affected_flows = set()
        for i in range(len(path) - 1):
            link = (path[i], path[i+1])
            if link in self.active_flows:
                # Calculate new bandwidth for this link immediately
                # link_capacity = self.link_bw_capacity[link]
                # num_flows = len(self.active_flows[link])
                new_link_bw = self.link_allocated_bw[link]

                for flow_state in self.active_flows[link]:
                    # Skip if this is the current process (don't interrupt ourselves!)
                    if flow_state.process != current_process and \
                       flow_state.process not in affected_flows: 
                        
                        # Innitalize decision flag
                        should_interrupt = False
                    
                        # Case 1: Bandwidth decreased (new flow joined)
                        # If the new link bandwidth is tighter than the flow's current bottleneck,
                        # it becomes the new bottleneck. We MUST interrupt.
                        if new_link_bw < flow_state.min_bandwidth:
                            should_interrupt = True
                            
                            
                        # Case 2: Bandwidth increased (e.g., flow left) or stayed high
                        # If the new bandwidth is higher than the current bottleneck,
                        # we only need to interrupt if this link WAS the bottleneck.
                        elif new_link_bw > flow_state.min_bandwidth:
                            # If link matches, it's this was the bottleneck
                            if flow_state.bottleneck_exact_link == link:
                                should_interrupt = True
                            # Else: This link is a "fat pipe" that got even fatter. No impact on flow.

                        if should_interrupt:
                            affected_flows.add(flow_state.process)
                            # Update the flow's new bandwidth info now for saving computing power
                            # flow_state.min_bandwidth = new_link_bw
                            # flow_state.bottleneck_exact_link == link
                            # Track which links caused this flow to be interrupted
                            # flow_state.changed_links.append(link)
        
        # Interrupt all affected processes (excluding ourselves)
        for process in affected_flows:
            process.interrupt()
    
    def _update_allocated_bandwidth(self, flow_state):
        """
        Calculate the bandwidth allocated to a flow based on fair sharing.
        Only recalculates for changed links to optimize performance.
        Returns tuple: (allocated_bandwidth, bottleneck_level)
        """
        path = flow_state.path
        # if not path or len(path) < 2:
        #     return float('inf'), None
        
        # If new flow, calculate min_bandwidth across all links
        if not flow_state.min_bandwidth:
            flow_state.min_bandwidth = float('inf')
            for i in range(len(path) - 1):
                link = (path[i], path[i+1])
                # link_capacity = self.link_bw_capacity[link]
                # num_flows = len(self.active_flows[link])
                
                per_flow_bw = self.link_allocated_bw[link]
                if per_flow_bw < flow_state.min_bandwidth:
                    flow_state.min_bandwidth = per_flow_bw
                    # NOTE: we save only bottleckneck_level first time (this is an assumption)
                    # but  bottleneck_exact_link is changed dynamically
                    flow_state.bottleneck_level = self.link_level[link]
                    flow_state.bottleneck_exact_link = link
        # Existing flow, only recalculate if any link changed
        else:
            # reset min_bandwidth to find new bottleneck
            flow_state.min_bandwidth = float('inf')

            for i in range(len(path) - 1):
                link = (path[i], path[i+1])    
                # link_capacity = self.link_bw_capacity[link]
                # num_flows = len(self.active_flows[link])
                
                per_flow_bw = self.link_allocated_bw[link]
                # Update if we found a smaller bandwidth
                if per_flow_bw < flow_state.min_bandwidth:
                    flow_state.min_bandwidth = per_flow_bw
                    flow_state.bottleneck_exact_link = link
                    # flow_state.bottleneck_level = self.link_level[link]
        
        # # Clear changed links after processing
        # flow_state.changed_links.clear()
        
        # return flow_state.min_bandwidth, flow_state.bottleneck_level
    
    # def _calculate_flow_delay(self, flow_state):
    #     """Calculate expected delay for a flow based on current bandwidth allocation."""
    #     # path = flow_state.path
        
    #     # Get allocated bandwidth and bottleneck info (reuses _get_allocated_bandwidth)
    #     allocated_bw, bottleneck_level = self._get_allocated_bandwidth(flow_state)
        
    #     # Calculate expected transmission time
    #     expected_delay = flow_state.data_remaining / allocated_bw
        
    #     return expected_delay, bottleneck_level


    def find_cluster(self, request):
        """Find appropriate clusters for request processing
           Returns: ( dict(cluster -> [path_direct, path_indirect]),
                     dict(cluster -> combined_failed_links) )
        """
        list_paths = {}
        found_direct = False
        found_indirect = False

        if variables.CLUSTER_STRATEGY == 'centralized_cloud':
            # direct to cloud
            found_direct, path_direct = self.find_possible_path(request.origin_node,
                self.clusters[variables.CENTRAL_CLOUD].node)
            # from cloud to data
            if found_direct:
                if request.data_path_required:
                    found_indirect, path_indirect = self.find_possible_path(
                        self.clusters[variables.CENTRAL_CLOUD].node,
                        request.data_node)
                else:
                    found_indirect = True
                    path_indirect = []
            # if both paths found, return them
            if found_direct and found_indirect:
                list_paths[self.clusters[variables.CENTRAL_CLOUD]] = [path_direct, path_indirect]
                # failed_links_map[self.clusters['cloud']] = combined
                return True, list_paths
            else:
                # merge failed‐links counts
                # combined = {}
                # for fl in (failed_direct or {}), (failed_indirect or {}):
                #     for lvl, cnt in fl.items():
                #         combined[lvl] = combined.get(lvl, 0) + cnt
                return False, {}
                    
        # elif CLUSTER_STRATEGY == 'distributed_cloud':
        else:
            nearby_clusters = self.graph.nodes[request.origin_node]['nearby_clusters']
            for cluster in nearby_clusters:
                # direct to cloud
                found_direct, path_direct = self.find_possible_path(
                    request.origin_node,
                    cluster.node
                )
                # from cloud to data
                if found_direct:
                    if request.data_path_required:
                        found_indirect, path_indirect = self.find_possible_path(
                            cluster.node,
                            request.data_node
                        )
                    else:
                        found_indirect = True
                        path_indirect = []
                # if both paths found, return them
                if found_direct and found_indirect:
                    list_paths[cluster] = [path_direct, path_indirect]
            # If no cluster could satisfy the request
            if not list_paths:
                return False, {}
            else:
                return True, list_paths
        # else:
        #     raise ValueError(f"Unknown topology strategy: {CLUSTER_STRATEGY}")
    
    def find_nearest_edge_cluster(self, src):
        """Find nearest edge cluster in terms of latency"""
        nearest_cluster = None
        nearest_cluster_node = None
        shortest_latency = float('inf')
        
        for edge in self.edge_cluster:
            edge_node = edge.node
            path = self.find_path(src, edge_node)
            if path:
                # Calculate total latency for this path
                total_latency = self.get_path_prop_delay(path)
                if total_latency < shortest_latency:
                    shortest_latency = total_latency
                    nearest_cluster = edge
                    nearest_cluster_node = edge_node
        
        return nearest_cluster, nearest_cluster_node

    # def get_edge_utilization(self, level):

    # Add this new method
    def get_cached_path(self, src, dst):
        """Retrieve a path from cache or compute and cache it if not found"""
        cache_key = (src, dst)
        
        # Check if path is in cache
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        else:
            return None
        # Path not in cache, compute it
        # path = self.find_hierachical_path(src, dst)
        # # Store in cache
        # self.path_cache[cache_key] = path

        # return path
    def save_cached_path(self, src, dst, path):
        """Save a computed path to the cache"""
        cache_key = (src, dst)
        self.path_cache[cache_key] = path
    # Add a similar cache for path latency

    def get_path_prop_delay(self, path):
        """Calculate the total latency of a path with caching.
        The cache is bidirectional; latency for a path is the same as for its reverse.
        """
        if not path or len(path) < 2:
            return 0
        
        # Create a canonical, immutable key for the cache.
        # This ensures that path (a,b,c) and (c,b,a) use the same cache entry.
        # We use the lexicographically smaller of the start/end nodes to decide the key's direction.
        if path[0] <= path[-1]:
            path_key = tuple(path)
        else:
            path_key = tuple(reversed(path))
        
        # Check if latency is cached using the canonical key
        if path_key in self.prop_delay_cache:
            return self.prop_delay_cache[path_key]
        
        # Else: Calculate latency for the original path direction
        total_latency = sum(self.graph.get_edge_data(path[i], path[i+1])['latency'] 
                           for i in range(len(path)-1))
        
        # Cache the result using the canonical key
        self.prop_delay_cache[path_key] = total_latency
        return total_latency
    
    def get_nearby(self, type, node, level=None):
        """Find and return a list of clusters sorted by proximity to the given node"""
        # Create a list to store clusters and their latencies
        sorted_clusters = []
        if type.startswith('massive'):
            if level == 1:
                parent = self.graph.nodes[self.graph.nodes[node]['parent']]['parent']
            elif level == 2:
                parent = self.graph.nodes[node]['parent']
            elif level == 3:
                parent = node
            else:
                raise ValueError(f"Edge cloud proximity search only supports level 1, 2 and 3")
            for cluster_name, cluster in self.edge_clusters.items():
                if cluster.node == parent:
                    sorted_clusters.append(cluster) 
                    break
        elif type.startswith('x_per_ring'):
            nearby_dc_nodes = self.node_dc_mapping.get(node, [])
            
            if len(nearby_dc_nodes) == 1:
                # Single DC, add it directly
                dc_node = nearby_dc_nodes[0]
                for cluster_name, cluster in self.edge_clusters.items():
                    if cluster.node == dc_node:
                        sorted_clusters.append(cluster)
                        break
            elif len(nearby_dc_nodes) > 1:
                # Multiple DCs, find the nearest by hop count
                min_hops = float('inf')
                nearest_clusters = []
                
                for dc_node in nearby_dc_nodes:
                    path = self.get_cached_path(node, dc_node)
                    if not path:
                        path = self.find_hierachical_path(node, dc_node)
                        if path:
                            self.save_cached_path(node, dc_node, path)
                    
                    if path:
                        hops = len(path) - 1
                        if hops < min_hops:
                            min_hops = hops
                            nearest_clusters = [dc_node]
                        elif hops == min_hops:
                            nearest_clusters.append(dc_node)
                
                # Pick one randomly if there are ties
                if nearest_clusters:
                    selected_dc = random.choice(nearest_clusters)
                    for cluster_name, cluster in self.edge_clusters.items():
                        if cluster.node == selected_dc:
                            sorted_clusters.append(cluster)
                            break
        else:
            raise ValueError(f"Unknown cluster type for proximity search: {type}")
        
         # Add central cloud for any edge_cloud strategy
        if 'cloud' in type:
            for cluster_name, cluster in self.cloud_clusters.items():
                if cluster_name == variables.CENTRAL_CLOUD:
                    sorted_clusters.append(cluster) 
                    break  
        return sorted_clusters 
    
    def equally(self, level):
        """Provision servers equally among network nodes."""
        level_nodes = [node for node, data in self.graph.nodes(data=True) if data.get('level') == level]
        num_level_nodes = len(level_nodes)

        servers_per_cluster = variables.EDGE_SERVER_NUMBER // num_level_nodes
        remaining_servers = variables.EDGE_SERVER_NUMBER % num_level_nodes
        
        edge_clusters = {}
        for i, node_name in enumerate(level_nodes):
            num_servers = servers_per_cluster + (1 if i < remaining_servers else 0)
            cluster_name = f"edge-{node_name}"
            config = variables.PC.copy()
            config.update({
                "name": cluster_name,
                "node": node_name,
                "num_servers": num_servers,
            })
            edge_clusters[cluster_name] = Cluster(self.env, config)
        
        return edge_clusters

    def population_massive(self, level):
        """Provision servers based on weighted population of network nodes."""
        level_nodes_data = {node: data['population'] for node, data in self.graph.nodes(data=True) if data.get('level') == level}
        total_population = sum(level_nodes_data.values())

        # Calculate initial server distribution based on population weight
        server_distribution = {node: (pop / total_population) * variables.EDGE_SERVER_NUMBER for node, pop in level_nodes_data.items()}
        
        # Allocate integer number of servers and track remainders
        allocated_servers = {node: int(dist) for node, dist in server_distribution.items()}
        remainders = {node: dist - int(dist) for node, dist in server_distribution.items()}
        
        # Distribute remaining servers based on largest remainders
        num_allocated = sum(allocated_servers.values())
        servers_to_distribute = variables.EDGE_SERVER_NUMBER - num_allocated
        
        # Sort nodes by remainder descending
        sorted_nodes_by_remainder = sorted(remainders, key=remainders.get, reverse=True)
        
        for i in range(servers_to_distribute):
            node_to_get_server = sorted_nodes_by_remainder[i]
            allocated_servers[node_to_get_server] += 1
            
        edge_clusters = {}
        for node_name, num_servers in allocated_servers.items():
            if num_servers > 0:
                cluster_name = f"edge-{node_name}"
                config = variables.PC.copy()
                config.update({
                    "name": cluster_name,
                    "node": node_name,
                    "num_servers": num_servers,
                })
                edge_clusters[cluster_name] = Cluster(self.env, config)
        
        return edge_clusters

    def pop_x_per_ring(self, level, number):
        """Provision servers with 'number' DCs per ring (parent group), selecting the highest population nodes.
        Server allocation is weighted based on the total population of each group, then divided equally among the DCs in that ring.
        
        For level=3: Assigns nodes within a group to 'number' DCs within the same group.
        For level=2: Assigns all level-3 nodes to 'number' level-2 DCs (their parent or parent's neighbors).
        """
        if level not in [2, 3]:
            raise ValueError(f"pop_x_per_ring strategy only supports level 2 or 3, got {level}")
        
        # Group nodes by their parent and calculate total population per group
        parent_groups = {}
        group_populations = {}
        
        for node, data in self.graph.nodes(data=True):
            if data.get('level') == level:
                parent = data.get('parent')
                if parent not in parent_groups:
                    parent_groups[parent] = []
                    group_populations[parent] = 0
                parent_groups[parent].append((node, data['population']))
                group_populations[parent] += data['population']
        
        # For each parent group, select top 'number' nodes with highest population
        selected_nodes_by_ring = {}  # {parent: [(node, group_population), ...]}
        for parent, nodes_list in parent_groups.items():
            # Sort nodes by population descending and take top 'number'
            sorted_nodes = sorted(nodes_list, key=lambda x: x[1], reverse=True)
            top_nodes = sorted_nodes[:min(number, len(sorted_nodes))]
            # Store selected nodes with their group's total population
            selected_nodes_by_ring[parent] = [(node, group_populations[parent]) for node, _ in top_nodes]
        
        # Calculate total population across all groups
        total_population = sum(group_populations.values())
        
        # Calculate server allocation per ring based on group population weight
        ring_server_allocation = {}
        ring_server_remainders = {}
        
        for parent, group_pop in group_populations.items():
            # Initial floating-point allocation for this ring
            allocation = (group_pop / total_population) * variables.EDGE_SERVER_NUMBER
            ring_server_allocation[parent] = int(allocation)
            ring_server_remainders[parent] = allocation - int(allocation)
        
        # Distribute remaining servers to rings based on largest remainders
        num_allocated = sum(ring_server_allocation.values())
        servers_to_distribute = variables.EDGE_SERVER_NUMBER - num_allocated
        
        sorted_rings_by_remainder = sorted(ring_server_remainders, key=ring_server_remainders.get, reverse=True)
        
        for i in range(servers_to_distribute):
            ring_to_get_server = sorted_rings_by_remainder[i]
            ring_server_allocation[ring_to_get_server] += 1
        
        # Now distribute each ring's servers equally among its selected nodes
        edge_clusters = {}
        for parent, nodes_list in selected_nodes_by_ring.items():
            total_servers_for_ring = ring_server_allocation[parent]
            num_nodes_in_ring = len(nodes_list)
            
            # Divide servers equally among nodes in this ring
            servers_per_node = total_servers_for_ring // num_nodes_in_ring
            remaining_servers = total_servers_for_ring % num_nodes_in_ring
            
            for i, (node_name, _) in enumerate(nodes_list):
                # Give extra server to first 'remaining_servers' nodes
                num_servers = servers_per_node + (1 if i < remaining_servers else 0)
                
                if num_servers > 0:
                    cluster_name = f"edge-{node_name}"
                    config = variables.PC.copy()
                    config.update({
                        "name": cluster_name,
                        "node": node_name,
                        "num_servers": num_servers,
                    })
                    edge_clusters[cluster_name] = Cluster(self.env, config)
        
        # === Build node_dc_mapping based on level ===
        if level == 3:
            # For level 3: Each level-3 node maps to DCs within its own group (same parent)
            for parent, dc_nodes_list in selected_nodes_by_ring.items():
                dc_node_ids = [dc_node for dc_node, _ in dc_nodes_list]
                
                # Extract all level-3 nodes from parent_groups
                all_level3_in_group = [node for node, _ in parent_groups[parent]]
                
                # Assign these DCs to all level-3 nodes in the group
                for node_id in all_level3_in_group:
                    self.node_dc_mapping[node_id] = dc_node_ids
        
        elif level == 2:
            # For level 2: Get all level-2 DCs
            dc_node_ids = [dc_node for nodes_list in selected_nodes_by_ring.values() 
                          for dc_node, _ in nodes_list]
            
            # For each grandparent group
            for grandparent in selected_nodes_by_ring.keys():
                # Extract all level-2 nodes from parent_groups
                all_level2_in_group = [node for node, _ in parent_groups[grandparent]]
                
                # Find which DCs belong to this group
                group_dc_nodes = [dc for dc in dc_node_ids if dc in all_level2_in_group]
                
                # Assign these DCs to all level-3 nodes whose parent is in this group
                for node, data in self.graph.nodes(data=True):
                    if data.get('level') == 3 and data.get('parent') in all_level2_in_group:
                        self.node_dc_mapping[node] = group_dc_nodes
        
        return edge_clusters

    def defined_edge_DCs(self, level, cluster_strategy, strategy='equally', number=1):
        # Find all nodes with level=1
        if strategy == 'equally':
            if "massive" in cluster_strategy:
                edge_clusters = self.equally(level)
            else:
                raise ValueError(f"Unknown edge DC provisioning cluster strategy for 'equally': {cluster_strategy}")
        elif strategy == 'population_weighted':
            if "massive" in cluster_strategy:
                edge_clusters = self.population_massive(level)
            elif 'x_per_ring' in cluster_strategy:
                edge_clusters = self.pop_x_per_ring(level, number)
            else:
                raise ValueError(f"Unknown edge DC provisioning cluster strategy for 'population_weighted': {cluster_strategy}")
        else:
            raise ValueError(f"Unknown edge DC provisioning strategy: {strategy}")

        if not edge_clusters:
            print(f"Warning: No level {level} nodes found to create edge clusters.")
            return {}

        print(f"Defined {len(edge_clusters)} edge clusters at node level {level} with '{strategy}' strategy.")
        return edge_clusters
