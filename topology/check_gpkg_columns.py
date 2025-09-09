import geopandas as gpd

# Path to the GeoPackage file
gpkg_path = "./topology/df_centroids.gpkg"  # Update this path if the file is in a different location

try:
    # Read the GeoPackage file
    gdf = gpd.read_file(gpkg_path)
    
    # Print column names
    print("Columns in the GeoPackage file:")
    for i, col in enumerate(gdf.columns):
        print(f"{i}: {col}")
    
    # Print a sample of the data (first few rows)
    print("\nSample data:")
    print(gdf.head())
    
    # Print information about the geometry
    print("\nGeometry information:")
    print(f"Geometry type: {gdf.geometry.geom_type.unique()}")
    print(f"CRS: {gdf.crs}")
    
except Exception as e:
    print(f"Error reading the GeoPackage file: {e}")
