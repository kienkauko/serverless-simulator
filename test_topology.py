import sys
import os
import networkx as nx
import traceback
from Topology_update import Topology_update

def main():
    # Set path to edge file (contains both node and edge data)
    edge_path = os.path.join('topology', 'edge.json')
    
    
    try:
        # Print path to make sure it's correct
        # print(f"Edge path: {os.path.abspath(edge_path)}")
        
        # Initialize topology
        print("Initializing topology...")
        
        topology = Topology_update(edge_path)
        
        # Print basic statistics
        msg = f"Topology loaded with {topology.graph.number_of_nodes()} nodes and {topology.graph.number_of_edges()} edges"
        # print(msg)
       
        # Print some edge information
        # msg = "\nSample edge information:"
        # print(msg)
        
        # edges = list(topology.graph.edges(data=True))[:5]
        # for u, v, data in edges:
        #     msg = f"Edge {u}-{v}: bandwidth={data['bandwidth']}, latency={data['latency']}"
        #     print(msg)
        
        # Test path finding between specific nodes
        test_pairs = [
            ('3326', '3311', "Path between")
        ]
        
        for src, dst, description in test_pairs:
            msg = f"\n{description}: {src} to {dst}"
            print(msg)
            
            if src in topology.graph.nodes:
                src_info = f"Source node {src} exists"
                print(src_info)
                src_location = topology.graph.nodes[src]['location']
                msg = f"Source location: {src_location}"
                print(msg)
            else:
                msg = f"Source node {src} not found in topology"
                print(msg)
            
            if dst in topology.graph.nodes:
                dst_info = f"Destination node {dst} exists"
                print(dst_info)
                dst_location = topology.graph.nodes[dst]['location']
                msg = f"Destination location: {dst_location}"
                print(msg)
            else:
                msg = f"Destination node {dst} not found in topology"
                print(msg)
            
            path = topology.find_path(src, dst)
            if path:
                msg = f"Path found: {path}"
                print(msg)
                
                msg = f"Path length: {len(path)} nodes, Path latency: {topology.get_path_latency(path):.4f} ms"
                print(msg)
            else:
                msg = "No path found"
                print(msg)
        
        print("Test completed successfully.")
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        print("Test failed.")

if __name__ == "__main__":
    main()
