#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.patches import Rectangle
import importlib.util
import sys
from variables import CLUSTER_CONFIG

def get_edge_cloud_nodes():
    """
    Get edge and cloud nodes from CLUSTER_CONFIG.
    Returns a tuple with (edge_node, cloud_node)
    """
    edge_nodes = []
    cloud_nodes = []
    
    for cluster_type, config in CLUSTER_CONFIG.items():
        if cluster_type == "edge":
            edge_nodes.append(config["node"])
        elif cluster_type == "cloud":
            cloud_nodes.append(config["node"])
    
    # Return the first edge and cloud nodes found
    # If any are missing, return empty strings
    edge_node = edge_nodes[0] if edge_nodes else ""
    cloud_node = cloud_nodes[0] if cloud_nodes else ""
    
    return edge_node, cloud_node

def read_topology_from_csv(file_path):
    """
    Read network topology from a CSV file.
    
    Each row in the CSV should contain:
    - Source node
    - Destination node
    - Bandwidth (Mbps)
    - Latency (ms)
    - Jitter (ms)
    
    Returns:
        A NetworkX graph object representing the topology
    """
    G = nx.Graph()
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            if len(row) >= 5:  # Ensure we have all required fields
                source = row[0]
                dest = row[1]
                bandwidth = float(row[2])
                latency = float(row[3])
                jitter = float(row[4])
                
                # Add nodes if they don't exist
                if source not in G:
                    G.add_node(source)
                if dest not in G:
                    G.add_node(dest)
                
                # Add edge with attributes
                G.add_edge(source, dest, 
                          bandwidth=bandwidth,
                          latency=latency,
                          jitter=jitter)
    
    return G

def visualize_topology(graph):
    """
    Visualize the network topology using matplotlib.
    
    Args:
        graph: A NetworkX graph object representing the topology
    """
    plt.figure(figsize=(12, 8))
    
    # Set up a nice layout for the graph
    pos = nx.spring_layout(graph, seed=42)
    
    # Get all edge and cloud nodes from the CLUSTER_CONFIG
    edge_nodes = []
    cloud_nodes = []
    
    for cluster_type, config in CLUSTER_CONFIG.items():
        if cluster_type == "edge":
            edge_nodes.append(config["node"])
        elif cluster_type == "cloud":
            cloud_nodes.append(config["node"])
    
    # Draw nodes with different colors for edge, cloud, and regular nodes
    node_colors = []
    for node in graph.nodes():
        if any(node.lower() == edge_node.lower() for edge_node in edge_nodes):
            node_colors.append('green')
        elif any(node.lower() == cloud_node.lower() for cloud_node in cloud_nodes):
            node_colors.append('orange')
        else:
            node_colors.append('skyblue')
    
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=node_colors, alpha=0.8)
    
    # Draw edges with width proportional to bandwidth
    edge_width = [graph[u][v]['bandwidth'] / 2000 for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')
    
    # Add edge labels for latency
    edge_labels = {(u, v): f"{graph[u][v]['latency']}ms" for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)
    
    # Add boxes with "Edge" and "Cloud" labels
    ax = plt.gca()
    
    # Add label boxes for each edge and cloud node
    for edge_node in edge_nodes:
        if edge_node in pos:
            x, y = pos[edge_node]
            color = 'green'
            # Draw a rectangle below the node
            offset = 0.08
            rect_width = 0.07
            rect_height = 0.04
            rect = Rectangle((x - rect_width/2, y - offset - rect_height), 
                           rect_width, rect_height, 
                           facecolor=color, alpha=0.8, zorder=0)
            ax.add_patch(rect)
            # Add text in the rectangle
            plt.text(x, y - offset - rect_height/2, "EDGE", 
                   fontsize=9, ha='center', va='center', color='black', 
                   fontweight='bold', zorder=5)
                   
    # Do the same for cloud nodes
    for cloud_node in cloud_nodes:
        if cloud_node in pos:
            x, y = pos[cloud_node]
            color = 'orange'
            # Draw a rectangle below the node
            offset = 0.08
            rect_width = 0.07
            rect_height = 0.04
            rect = Rectangle((x - rect_width/2, y - offset - rect_height), 
                           rect_width, rect_height, 
                           facecolor=color, alpha=0.8, zorder=0)
            ax.add_patch(rect)
            # Add text in the rectangle
            plt.text(x, y - offset - rect_height/2, "CLOUD", 
                   fontsize=9, ha='center', va='center', color='black', 
                   fontweight='bold', zorder=5)
    
    # Title and grid
    plt.title("Network Topology Visualization", fontsize=16)
    plt.grid(False)
    plt.axis('off')
    
    # Add a legend
    plt.text(0.02, 0.02, "Edge width: proportional to bandwidth\nEdge label: latency (ms)\nGreen node: Edge cluster\nOrange node: Cloud cluster", 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    return plt

def show_edge_info(graph):
    """
    Print information about each edge in the graph.
    
    Args:
        graph: A NetworkX graph object representing the topology
    """
    print("\nNetwork Link Information:")
    print("-" * 80)
    print(f"{'Source':^10} | {'Destination':^10} | {'Bandwidth':^10} | {'Latency':^10} | {'Jitter':^10}")
    print("-" * 80)
    
    for u, v, data in graph.edges(data=True):
        print(f"{u:^10} | {v:^10} | {data['bandwidth']:^10.0f} | {data['latency']:^10.1f} | {data['jitter']:^10.1f}")

def main():
    # Determine the absolute path to the CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'topology.csv')
    
    try:
        # Read the topology from the CSV file
        G = read_topology_from_csv(csv_path)
        
        if len(G.nodes()) == 0:
            print(f"No valid topology data found in {csv_path}")
            return
        
        print(f"Successfully read topology with {len(G.nodes())} nodes and {len(G.edges())} links.")
        
        # Show link information
        show_edge_info(G)
        
        # Visualize the topology
        plt = visualize_topology(G)
        
        # Get network statistics
        print("\nNetwork Statistics:")
        print(f"Network Diameter: {nx.diameter(G)}")
        print(f"Average Shortest Path Length: {nx.average_shortest_path_length(G):.3f}")
        print(f"Average Node Degree: {sum(dict(G.degree()).values()) / len(G):.2f}")
        
        # Show the visualization
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
