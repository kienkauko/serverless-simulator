import networkx as nx
import json
import math
from pyproj import Transformer
from Cluster import Cluster
import variables

class Topology:
    def __init__(self, env, edge_path, cluster_info, network_model='reservation'):
        # self.clusters = clusters
        # Store edge cluster for easy access if provided
        # self.edge_cluster = [cluster for cluster_name, cluster in clusters.items() if cluster_name == "edge"] if clusters else []
        print(f"Initializing topology with network model: {network_model}...")
        self.network_model = network_model
        # Initialize graph
        self.graph = nx.DiGraph()
        self.env = env
        # Add a path cache
        self.path_cache = {}  # Format: (src, dst) -> path
        self.path_latency_cache = {}  # Format: (tuple(path)) -> latency

        # Initialize clusters
        self.clusters = {}
        self.cloud_clusters = {}
        self.edge_clusters = {}

        # Load edge data from JSON (includes both node and link data)
        with open(edge_path, 'r') as f:
            edge_data = json.load(f)
        
        # Process nodes from edge.json
        for node in edge_data.get('nodes', []):
            node_name = node['name']
            node_id = node_name.replace('_R0', '')  # Remove _R0 suffix
            # Add node to the graph with all attributes
            self.graph.add_node(node_id, 
                                location=node['location'],
                                level=node['level'],
                                population=node['population'],
                                parent=str(node.get('parent', None)),
                                nearby_clusters=None)  # Will be updated later if needed
        
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
                # Calculate latency based on node locations
                latency = self.calculate_latency(n1_id, n2_id)
                
                # Get bandwidth from JSON or use default
                # Current bandwidth is: 3: 10, 2: 40, 1: 100, 0: 400 Gbps
                bandwidth = float(link.get('bandwidth', 1000000000.0))  # Default: 1 Gbps
                
                # Get link level
                node1_level = self.graph.nodes[n1_id]['level']
                node2_level = self.graph.nodes[n2_id]['level']
                combined_level = f"{node1_level}-{node2_level}"
                
                # Add edge to the graph with attributes based on the network model
                if self.network_model == 'reservation':
                    self.graph.add_edge(n1_id, n2_id, 
                                       bandwidth=bandwidth, 
                                       available_bandwidth=bandwidth, 
                                       latency=latency,
                                       level=combined_level)
                elif self.network_model == 'ps':
                    self.graph.add_edge(n1_id, n2_id, 
                                       bandwidth=bandwidth, 
                                       num_active_flows=0, 
                                       latency=latency,
                                       level=combined_level)

        # Load cluster information from JSON
        with open(cluster_info, 'r') as f:
            cluster_data = json.load(f)
        for cluster_name, config in cluster_data.items():
            # Create the cluster with its servers
            if variables.CLUSTER_STRATEGY != "distributed_cloud":
                cluster_name = variables.CENTRAL_CLOUD
                self.clusters[cluster_name] = Cluster(self.env, cluster_name, config)
                self.cloud_clusters[cluster_name] = self.clusters[cluster_name]
                break  # Only one cluster needed
            else:
                self.clusters[cluster_name] = Cluster(self.env, cluster_name, config)
            # Store cloud clusters separately for later usage
            # NOTE: these logics are not very useful in current implementation
            if cluster_name.startswith("cloud"):
                self.cloud_clusters[cluster_name] = self.clusters[cluster_name]

        # Initialize nearby clusters for each node
        if variables.CLUSTER_STRATEGY != "centralized_cloud":
            print("Calculating nearby clusters for each node based on strategy: ", variables.CLUSTER_STRATEGY)
            if variables.CLUSTER_STRATEGY.startswith("massive_edge"):
                self.edge_clusters = self.defined_edge_DCs(variables.EDGE_DC_LEVEL, variables.EDGE_SERVER_PROVISION_STRATEGY)
                self.clusters.update(self.edge_clusters)
            for node_id in [n for n, data in self.graph.nodes(data=True) if data.get('level') == 3]:
                nearby_clusters = self.get_nearby(variables.CLUSTER_STRATEGY, node_id, variables.EDGE_DC_LEVEL)
                self.graph.nodes[node_id]['nearby_clusters'] = nearby_clusters

        print("Topology initialized.")
        
    def get_link_utilization(self):
        """Calculate and return average link utilization statistics grouped by level."""
        utilization_stats = {}
        level_edges = {}
        
        # Collect utilization data for each edge grouped by level
        for n1, n2, data in self.graph.edges(data=True):
            bandwidth = data.get('bandwidth', 0)
            available_bw = data.get('available_bandwidth', 0)
            level = data.get('level', 'unknown')
            
            # Calculate utilization as percentage
            utilization = ((bandwidth - available_bw) / bandwidth) * 100 
            
            if level not in level_edges:
                level_edges[level] = {
                    'count': 0,
                    'total_utilization': 0
                }
            
            level_edges[level]['count'] += 1
            level_edges[level]['total_utilization'] += utilization
        
        # Calculate average utilization for each level
        for level, stats in level_edges.items():
            if stats['count'] > 0:
                utilization_stats[level] = stats['total_utilization'] / stats['count']
        
        return utilization_stats
    
    def calculate_latency(self, node1, node2):
        """Calculate latency between two nodes based on their locations in the JSON file."""
        # Get locations in EPSG:3857
        loc1 = self.graph.nodes[node1]['location']
        loc2 = self.graph.nodes[node2]['location']
        
        # Convert from EPSG:3857 to EPSG:4326 (from Web Mercator to standard lon/lat)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon1, lat1 = transformer.transform(loc1[0], loc1[1])
        lon2, lat2 = transformer.transform(loc2[0], loc2[1])
        
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
        
        return latency_ms
    
    # def find_path(self, src, dst, required_bw=None):
    #     """Find shortest path between source and destination nodes."""
    #     # Convert IDs to strings for consistency
    #     src = str(src)
    #     dst = str(dst)
        
    #     # If source and destination are the same
    #     if src == dst:
    #         return [src]
        
    #     # Find the shortest path considering bandwidth if specified
    #     return self.find_shortest_path(src, dst, required_bw)

        # if path:
        #     return path
        # else:

    
    def find_possible_path(self, src, dst, required_bw=None):
        """Find shortest path between two nodes considering bandwidth if specified."""
        # try:
        # if required_bw is None:
        #     # Use simple shortest path when bandwidth is not a constraint
        #     path = nx.shortest_path(self.graph, src, dst, weight='latency')
        #     return path
        
        # Find paths ordered by latency
        # paths = list(nx.shortest_simple_paths(self.graph, src, dst, weight='latency'))
        path = self.get_cached_path(src, dst)
        if not path:
            path = self.find_hierachical_path(src, dst)
            
        # Track failed links by level
        failed_links_by_level = {}
        
        # Check bandwidth constraints if path exists
        if path:
            self.save_cached_path(src, dst, path)
            if self.network_model == 'reservation':
                for i in range(len(path)-1):
                    edge = self.graph.get_edge_data(path[i], path[i+1])
                    if edge['available_bandwidth'] < required_bw:
                        # Record the level of the failed link
                        level = edge.get('level', 'unknown')
                        if level not in failed_links_by_level:
                            failed_links_by_level[level] = 0
                        failed_links_by_level[level] += 1
                        return False, None, failed_links_by_level
                return True, path, failed_links_by_level
            elif self.network_model == 'ps':
                # In PS model, we don't reject based on bandwidth.
                return True, path, failed_links_by_level
        
        # if failed_links_by_level.empty():
        #     return True, None, failed_links_by_level
        else:
            return False, None, failed_links_by_level
            
        # except nx.NetworkXNoPath:
        #     return False, None, failed_links_by_level

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
            # node_b_parent = self.graph.nodes[node_b].get('parent', None)

            # node_a_parent_level = self.graph.nodes[node_a_parent]['level'] 
            # node_b_parent_level = self.graph.nodes[node_b_parent]['level']

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
         
    # def get_path_latency(self, path):
    #     """Calculate the total latency of a path"""
    #     if not path or len(path) < 2:
    #         return 0
            
    #     total_latency = sum(self.graph.get_edge_data(path[i], path[i+1])['latency'] 
    #                         for i in range(len(path)-1))
    #     return total_latency
    
    def implement_path(self, path, bw):
        """Reserve bandwidth along a path or increment flow count."""
        if not path or len(path) < 2:
            return
            
        for i in range(len(path)-1):
            if self.network_model == 'reservation':
                self.graph[path[i]][path[i+1]]['available_bandwidth'] -= bw
            elif self.network_model == 'ps':
                self.graph[path[i]][path[i+1]]['num_active_flows'] += 1

    def release_path(self, path, bw):
        """Release bandwidth along a path or decrement flow count."""
        if not path or len(path) < 2:
            return
            
        for i in range(len(path)-1):
            if self.network_model == 'reservation':
                if bw is not None:
                    self.graph[path[i]][path[i+1]]['available_bandwidth'] += bw
            elif self.network_model == 'ps':
                self.graph[path[i]][path[i+1]]['num_active_flows'] -= 1
    
    def get_path_transmission_delay(self, path, packet_size):
        """
        Calculate the total transmission delay of a path for the PS model.
        This version models pipelining by finding the bottleneck link.
        The total delay is determined by the slowest link in the path.
        Returns a tuple: (total_transmission_delay, bottleneck_level, delay_by_level).
        """
        if not path or len(path) < 2 or packet_size is None:
            return 0, None, {}
        
        bottleneck_bandwidth = float('inf')
        bottleneck_level = None
        delay_by_level = {} # This will represent the delay contribution if this level were the bottleneck

        # First, find the bottleneck bandwidth for the new flow along the path
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            link_capacity = edge_data['bandwidth']
            # Add 1 to num_flows to account for the current request
            num_flows = edge_data['num_active_flows'] + 1
            
            flow_bandwidth = link_capacity / num_flows
            
            if flow_bandwidth < bottleneck_bandwidth:
                bottleneck_bandwidth = flow_bandwidth
                bottleneck_level = edge_data.get('level', 'unknown')

        # Calculate total transmission delay based on the single bottleneck link
        if bottleneck_bandwidth > 0:
            total_transmission_delay = packet_size / bottleneck_bandwidth
        else:
            total_transmission_delay = float('inf')

        # For logging/analysis, we can still calculate what the delay would be per level
        # Note: This is for analysis only, these delays are not summed.
        # for i in range(len(path) - 1):
        #     edge_data = self.graph.get_edge_data(path[i], path[i+1])
        #     link_capacity = edge_data['bandwidth']
        #     num_flows = edge_data['num_active_flows'] + 1
        #     flow_bandwidth = link_capacity / num_flows
        #     link_delay = packet_size / flow_bandwidth
        #     level = edge_data.get('level', 'unknown')
            # delay_by_level[level] = delay_by_level.get(level, 0) + link_delay

        return total_transmission_delay, bottleneck_level

    def update_request_delay(self, request, paths, type='upload'):
        """Calculate propagation and transmission delays for the given paths."""
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
        
        # NOTE: current reservation is behind updates -> it's wrong
        if self.network_model == 'reservation':
            prop_delay = 0
            if request.data_path_required:
                prop_delay = self.get_path_latency(path_direct) + self.get_path_latency(path_indirect) * 2  # Round trip
            else:
                prop_delay = self.get_path_latency(path_direct) * 2
            # Set delays variables
            request.prop_delay += prop_delay / 1000  # Convert ms to seconds
            request.network_delay += request.prop_delay  

        else:   # PS model
            # Static propagation delay
            prop_delay = self.get_path_latency(path_direct)
            # Dynamic transmission delay and returns bottleneck info
            trans_delay_direct, bottleneck_direct = self.get_path_transmission_delay(path_direct, packet_size_direct)
            trans_delay_indirect, bottleneck_indirect = 0, None

            if request.data_path_required:
                prop_delay += self.get_path_latency(path_indirect)
                trans_delay_indirect, bottleneck_indirect = self.get_path_transmission_delay(path_indirect, packet_size_indirect)

            request.prop_delay += prop_delay / 1000  # Convert ms to seconds
            
            # Store bottleneck info on the request
            max_trans_delay = max(trans_delay_direct, trans_delay_indirect)
            if max_trans_delay > request.max_trans_delay:
                request.max_trans_delay = max_trans_delay
                if max_trans_delay ==  trans_delay_direct:
                    request.bottleneck = bottleneck_direct
                else:
                    request.bottleneck = bottleneck_indirect
            # request.delay_by_level_direct = delay_by_level_direct
            # request.delay_by_level_indirect = delay_by_level_indirect

            # Total network delay
            request.network_delay += (request.prop_delay + trans_delay_direct + trans_delay_indirect)

    def make_paths(self, request, paths):
        """Implement bandwidth reservation or flow counting for the given paths."""
        if self.network_model == 'reservation':
            self.implement_path(paths[0], request.bandwidth_direct)
            if request.data_path_required:
                self.implement_path(paths[1], request.bandwidth_indirect)
        elif self.network_model == 'ps':
            self.implement_path(paths[0], None)
            if request.data_path_required:
                self.implement_path(paths[1], None)


    def remove_paths(self, request, paths):
        """Release bandwidth for paths"""
        if self.network_model == 'reservation':
            self.release_path(paths[0], request.bandwidth_direct)
            if request.data_path_required:
                self.release_path(paths[1], request.bandwidth_indirect)
        elif self.network_model == 'ps':
            self.release_path(paths[0], None)
            if request.data_path_required:
                self.release_path(paths[1], None)
    
    def find_cluster(self, request):
        """Find appropriate clusters for request processing
           Returns: ( dict(cluster -> [path_direct, path_indirect]),
                     dict(cluster -> combined_failed_links) )
        """
        list_paths = {}
        failed_links_map = {}
        found_direct = False
        found_indirect = False

        if variables.CLUSTER_STRATEGY == 'centralized_cloud':
            # direct to cloud
            found_direct, path_direct, failed_links_map = self.find_possible_path(
                request.origin_node,
                self.clusters[variables.CENTRAL_CLOUD].node,
                request.bandwidth_direct
            )
            # from cloud to data
            if found_direct:
                if request.data_path_required:
                    found_indirect, path_indirect, failed_links_map = self.find_possible_path(
                        self.clusters[variables.CENTRAL_CLOUD].node,
                        request.data_node,
                        request.bandwidth_indirect
                    )
                else:
                    found_indirect = True
                    path_indirect = []
            # if both paths found, return them
            if found_direct and found_indirect:
                list_paths[self.clusters[variables.CENTRAL_CLOUD]] = [path_direct, path_indirect]
                # failed_links_map[self.clusters['cloud']] = combined
                return True, list_paths, {}
            else:
                # merge failed‐links counts
                # combined = {}
                # for fl in (failed_direct or {}), (failed_indirect or {}):
                #     for lvl, cnt in fl.items():
                #         combined[lvl] = combined.get(lvl, 0) + cnt
                return False, {}, failed_links_map
                    
        # elif CLUSTER_STRATEGY == 'distributed_cloud':
        else:
            nearby_clusters = self.graph.nodes[request.origin_node]['nearby_clusters']
            for cluster in nearby_clusters:
                # direct to cloud
                found_direct, path_direct, failed_links_map = self.find_possible_path(
                    request.origin_node,
                    cluster.node,
                    request.bandwidth_direct
                )
                # from cloud to data
                if found_direct:
                    if request.data_path_required:
                        found_indirect, path_indirect, failed_links_map = self.find_possible_path(
                            cluster.node,
                            request.data_node,
                            request.bandwidth_indirect
                        )
                    else:
                        found_indirect = True
                        path_indirect = []
                # if both paths found, return them
                if found_direct and found_indirect:
                    list_paths[cluster] = [path_direct, path_indirect]
            # If no cluster could satisfy the request
            if not list_paths:
                return False, {}, failed_links_map
            else:
                return True, list_paths, failed_links_map
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
                total_latency = self.get_path_latency(path)
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

    def get_path_latency(self, path):
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
        if path_key in self.path_latency_cache:
            return self.path_latency_cache[path_key]
        
        # Else: Calculate latency for the original path direction
        total_latency = sum(self.graph.get_edge_data(path[i], path[i+1])['latency'] 
                           for i in range(len(path)-1))
        
        # Cache the result using the canonical key
        self.path_latency_cache[path_key] = total_latency
        return total_latency
    
    def get_nearby(self, type, node, level=None):
        """Find and return a list of clusters sorted by proximity to the given node"""
        # Create a list to store clusters and their latencies
        sorted_clusters = []
        if type == 'distributed_cloud':
            clusters_with_latencies = []
            # Check proximity to all cloud clusters
            for cluster_name, cluster in self.cloud_clusters.items():                
                path = self.get_cached_path(node, cluster.node)
                if not path:
                    path = self.find_hierachical_path(node, cluster.node)
                if path:
                    self.save_cached_path(node, cluster.node, path)
                    total_latency = self.get_path_latency(path)
                    clusters_with_latencies.append((cluster, total_latency))
            
            # Sort clusters by latency (proximity)
            sorted_clusters = [c for c, _ in sorted(clusters_with_latencies, key=lambda x: x[1])]
            return sorted_clusters
        elif type.startswith('massive_edge'):
            if level == 1:
                parent = self.graph.nodes[self.graph.nodes[node]['parent']]['parent']
            elif level == 2:
                parent = self.graph.nodes[node]['parent']
            else:
                raise ValueError(f"Edge cloud proximity search only supports level 1 or 2")
                # path = self.get_cached_path(node, grandparent)
                # Find the edge cluster that corresponds to the grandparent node
                # edge_cluster = None
                # cloud_cluster = None
            for cluster_name, cluster in self.edge_clusters.items():
                if cluster.node == parent:
                    sorted_clusters.append(cluster) 
                    break
            # Add central cloud for massive_edge_cloud strategy
            if type == 'massive_edge_cloud':
                for cluster_name, cluster in self.cloud_clusters.items():
                    if cluster_name == variables.CENTRAL_CLOUD:
                        sorted_clusters.append(cluster) 
                        break  
            return sorted_clusters                             
        else:
            raise ValueError(f"Unknown cluster type for proximity search: {type}")
    
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
            config = {
                "node": node_name,
                "num_servers": num_servers,
                "server_cpu": 100.0,
                "server_ram": 100.0,
                "power_max": 60,
                "power_min": 10
            }
            edge_clusters[cluster_name] = Cluster(self.env, cluster_name, config)
        
        return edge_clusters

    def population_weighted(self, level):
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
                config = {
                    "node": node_name,
                    "num_servers": num_servers,
                    "server_cpu": 100.0,
                    "server_ram": 100.0,
                    "power_max": 60,
                    "power_min": 10
                }
                edge_clusters[cluster_name] = Cluster(self.env, cluster_name, config)
        
        return edge_clusters

    def defined_edge_DCs(self, level, strategy='equally'):
        # Find all nodes with level=1
        if strategy == 'equally':
            edge_clusters = self.equally(level)
        elif strategy == 'population_weighted':
            edge_clusters = self.population_weighted(level)
        else:
            raise ValueError(f"Unknown edge DC provisioning strategy: {strategy}")

        if not edge_clusters:
            print(f"Warning: No level {level} nodes found to create edge clusters.")
            return {}

        print(f"Defined {len(edge_clusters)} edge clusters at node level {level} with '{strategy}' strategy.")
        return edge_clusters
