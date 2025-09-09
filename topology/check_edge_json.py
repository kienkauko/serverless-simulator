import json
import os

def main():
    # Path to the edge.json file
    edge_json_path = os.path.join("topology", "edge.json")
    
    try:
        # Read the edge.json file
        with open(edge_json_path, 'r') as f:
            edge_data = json.load(f)
        
        # Check how many nodes have parent IDs
        nodes_with_parent = sum(1 for node in edge_data["nodes"] if "parent" in node)
        total_nodes = len(edge_data["nodes"])
        
        print(f"Total nodes: {total_nodes}")
        print(f"Nodes with parent IDs: {nodes_with_parent}")
        
        # Print example nodes (first 5) to check the format
        print("\nExample nodes:")
        for i, node in enumerate(edge_data["nodes"][:5]):
            print(f"{i+1}: {node}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
