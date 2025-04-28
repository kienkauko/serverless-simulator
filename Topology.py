import networkx as nx
import random  # Added for jitter calculation

class Topology:
    def __init__(self, file_path):
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

    def implement_path(self, path, bw):
        for i in range(len(path)-1):
            self.graph[path[i]][path[i+1]]['available_bandwidth'] -= bw

    def release_path(self, path, bw):
        for i in range(len(path)-1):
            self.graph[path[i]][path[i+1]]['available_bandwidth'] += bw
