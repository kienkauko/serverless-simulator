import networkx as nx
import random  # Added for jitter calculation

class Topology:
    def __init__(self, file_path, clusters=None):
        self.clusters = clusters
        # store edge cluster for easy access
        self.edge_cluster = [cluster for cluster_name, cluster in clusters.items() if cluster_name == "edge"]
        self.graph = nx.Graph()
        # Expect each line: node1,node2,bandwidth,latency,jitter
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 5:
                    continue
                node1, node2 = parts[0], parts[1]
                bw = float(parts[2])
                mean_latency = float(parts[3])
                jitter = float(parts[4])
                # Compute effective latency using Gaussian noise
                effective_latency = max(0, random.gauss(mean_latency, jitter))
                self.graph.add_edge(node1, node2, bandwidth=bw, available_bandwidth=bw, latency=effective_latency)

    def find_path(self, src, dst, required_bw):
        try:
            # NOTE: Here routing strategy is given 
            paths = nx.shortest_simple_paths(self.graph, src, dst, weight='latency')
        except nx.NetworkXNoPath:
            return None
        
        for path in paths:
            valid = True
            for i in range(len(path)-1):
                edge = self.graph.get_edge_data(path[i], path[i+1])
                if edge['available_bandwidth'] < required_bw:
                    valid = False
                    break
            if valid:
                # for i in range(len(path)-1):
                #     self.graph[path[i]][path[i+1]]['available_bandwidth'] -= required_bw
                return path
        return None

    # NOTE: simple_path doesnt consider bandwidth, just shortest path
    # Use it when bw is flexible (e.g., can be squeezed)
    def find_simple_path(self, src, dst):
        try:
            # NOTE: Here routing strategy is given 
            paths = nx.shortest_simple_paths(self.graph, src, dst, weight='latency')
            # Return only the first path (shortest)
            return next(paths)
        except nx.NetworkXNoPath:
            return None
    
    # Find nearest edge cluster in terms of latency
    def find_nearest_edge_cluster(self, src):
        nearest_cluster = None
        nearest_cluster_node = None
        shortest_latency = float('inf')
        
        for edge in self.edge_cluster:
            edge_node = edge.node
            path = self.find_simple_path(src, edge_node)
            if path:
                # Calculate total latency for this path
                total_latency = sum(self.graph.get_edge_data(path[i], path[i+1])['latency'] 
                                    for i in range(len(path)-1))
                if total_latency < shortest_latency:
                    shortest_latency = total_latency
                    nearest_cluster = edge
                    nearest_cluster_node = edge_node
        
        return nearest_cluster, nearest_cluster_node
    
    def get_path_latency(self, path):
        """Calculate the total latency of a path"""
        if not path:
            return 0
        total_latency = sum(self.graph.get_edge_data(path[i], path[i+1])['latency'] 
                            for i in range(len(path)-1))
        return total_latency

    def make_paths(self, request, paths):
        # Setup bandwidth and propagation delay for the request
        self.implement_path(paths[0], request.bandwidth_direct)
        request.prop_delay += self.get_path_latency(paths[0])*2  # Round trip
        self.implement_path(paths[1], request.bandwidth_indirect)
        request.prop_delay += self.get_path_latency(paths[1])
    
    def remove_paths(self, request, paths):
        self.release_path(paths[0], request.bandwidth_direct)
        self.release_path(paths[1], request.bandwidth_indirect)

    def implement_path(self, path, bw):
        for i in range(len(path)-1):
            # Double check bandwidth availability
            if self.graph.get_edge_data(path[i], path[i+1])['available_bandwidth'] < bw:
                raise ValueError(f"Insufficient bandwidth on edge {path[i]}-{path[i+1]} for {bw} units")
                        
            self.graph[path[i]][path[i+1]]['available_bandwidth'] -= bw

    def release_path(self, path, bw):
        for i in range(len(path)-1):
            self.graph[path[i]][path[i+1]]['available_bandwidth'] += bw

    
    def find_cluster(self, request, use_topo, strategy='always_cloud'):
        # NOTE: always add DC into the return list
        # Its job is to return a list of viable clusters (DCs or Edge DCs) for the request
        # Different strategies can be implemented here
        if use_topo:
            list_paths = {}
            paths = []
            if strategy == 'always_cloud':
                path_direct = self.find_path(request.origin_node, self.clusters['cloud'].node, request.bandwidth_direct)
                path_indirect = self.find_path(self.clusters['cloud'].node, request.data_node, request.bandwidth_indirect)
                paths.append(path_direct)
                paths.append(path_indirect)
                list_paths[self.clusters['cloud']] = paths
                return list_paths
            elif strategy == 'local_edge_then_cloud':
                destine_node = None
                if not request.local_edge_cluster:
                    # Find the nearest edge cluster to the request's origin node
                    edge_cluster, destine_node = self.find_nearest_edge_cluster(request.origin_node)
                else:
                    destine_node = request.local_edge_cluster
                path_direct = self.find_path(request.origin_node, destine_node, request.bandwidth_direct)
                path_indirect = self.find_path(destine_node, request.data_node, request.bandwidth_indirect)
                paths.append(path_direct)
                paths.append(path_indirect)
                list_paths[edge_cluster] = paths
                # Always add cloud as fallback
                paths = []
                path_direct = self.find_path(request.origin_node, self.clusters['cloud'].node, request.bandwidth_direct)
                path_indirect = self.find_path(self.clusters['cloud'].node, request.data_node, request.bandwidth_indirect)
                paths.append(path_direct)
                paths.append(path_indirect)
                list_paths[self.clusters['cloud']] = paths
                return list_paths
            else:
                raise ValueError(f"Unknown topology strategy: {strategy}")
 
        #     for cluster_name, cluster in self.clusters.items():
        #         path = self.topology.find_path(request.origin_node, cluster.node, bandwidth_demand)
        #         if path:
        #             paths[cluster_name] = path
        #             target_clusters.append(cluster_name)
                    
        #             # Calculate prop delay for this path (for metrics)
        #             prop = sum(self.topology.graph.get_edge_data(path[i], path[i+1])['latency'] 
        #                         for i in range(len(path)-1))
        #             # We'll set this on the request if this cluster is chosen
        #             request.potential_prop_delays[cluster_name] = 2 * prop
            
        #     if not target_clusters:
        #         print(f"{self.env.now:.2f} - BLOCK: No feasible path from {request.origin_node} to any cluster for {request}")
        #         request_stats['blocked_no_path'] += 1
        #         app_stats[request.app_id]['blocked_no_path'] += 1
        #         return
                
        #     # Sort target_clusters by propagation delay (lowest first)
        #     target_clusters.sort(key=lambda cluster_name: request.potential_prop_delays[cluster_name])
        #     print(f"{self.env.now:.2f} - Ordered clusters by propagation delay for {request}: {target_clusters}")
        # else:
        #     # No topology routing - all clusters are viable targets
        #     target_clusters = list(self.clusters.keys())
        #     # Set propagation delay to 0 for all clusters
        #     for cluster_name in target_clusters:
        #         request.potential_prop_delays[cluster_name] = 0.0