import geopandas as gpd
import json
import re
import os
import pandas as pd
import numpy as np

def main():
    # Path to the GeoPackage file
    gpkg_path = "./topology/df_centroids.gpkg"
    
    # Path to the edge.json file
    edge_json_path = os.path.join("topology", "edge.json")
    
    print(f"Reading GeoPackage file: {gpkg_path}")
    try:
        # Read the GeoPackage file
        gdf = gpd.read_file(gpkg_path)
        
        # Check if 'id' and 'parent_id' columns exist
        required_columns = ['id', 'parent_id']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        
        if missing_columns:
            print(f"Error: GeoPackage file is missing required columns: {missing_columns}")
            print("Available columns:", gdf.columns.tolist())
            return
        
        # Create a dictionary mapping node IDs to parent IDs, skipping NaN values
        node_parent_map = {}
        for idx, row in gdf.iterrows():
            if pd.notna(row['id']) and pd.notna(row['parent_id']):
                node_parent_map[int(row['id'])] = int(row['parent_id'])
        
        print(f"Found {len(node_parent_map)} nodes with valid parent IDs")
        
        # Read the edge.json file
        with open(edge_json_path, 'r') as f:
            edge_data = json.load(f)
        
        # Update the nodes with parent IDs
        updated_count = 0
        not_found_count = 0
        not_found_ids = []
        
        for node in edge_data["nodes"]:
            # Extract the node ID by removing "_R0" suffix
            node_name = node["name"]
            match = re.match(r"(\d+)_R0", node_name)
            
            if match:
                node_id = int(match.group(1))
                
                # Add parent_id to the node
                if node_id in node_parent_map:
                    node["parent"] = node_parent_map[node_id]
                    updated_count += 1
                else:
                    not_found_count += 1
                    not_found_ids.append(node_id)
        
        print(f"Updated {updated_count} nodes with parent IDs")
        if not_found_count > 0:
            print(f"Warning: Could not find parent IDs for {not_found_count} nodes")
            print(f"First 10 IDs not found: {not_found_ids[:10]}")
        
        # Write the updated edge.json file
        with open(edge_json_path, 'w') as f:
            json.dump(edge_data, f, indent=4)
        
        print(f"Updated edge.json file saved to {edge_json_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
