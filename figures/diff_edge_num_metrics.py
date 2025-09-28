import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the data from Excel file
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results.xlsx')
df = pd.read_excel(file_path)

# Include edge_cloud_level_1, edge_only_level_1, and centralized_cloud scenarios
filtered_df = df[df['cluster_strategy'].isin(['edge_cloud_level_1', 'edge_only_level_1', 'centralized_cloud'])]

# Define metrics to plot
metrics = [
    'blocking_percentage',
    'avg_offloaded_to_cloud',
    'avg_total_latency',
    'avg_spawn_time',
    'avg_processing_time',
    'avg_network_time'
]

# Get unique edge server numbers
edge_server_numbers = filtered_df['edge_server_number'].unique()

# Create a figure for each metric
plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    
    for edge_num in edge_server_numbers:
        # Filter data for the current edge server number
        edge_num_df = filtered_df[filtered_df['edge_server_number'] == edge_num]
        
        # Plot for edge_cloud_level_1
        ec_df = edge_num_df[edge_num_df['cluster_strategy'] == 'edge_cloud_level_1']
        if not ec_df.empty:
            ec_df = ec_df.sort_values(by='traffic_intensity')
            plt.plot(ec_df['traffic_intensity'], ec_df[metric], marker='o', 
                     label=f'Edge-Cloud ({edge_num} edges)')
        
        # Plot for edge_only_level_1
        eo_df = edge_num_df[edge_num_df['cluster_strategy'] == 'edge_only_level_1']
        if not eo_df.empty:
            eo_df = eo_df.sort_values(by='traffic_intensity')
            plt.plot(eo_df['traffic_intensity'], eo_df[metric], marker='s', 
                     linestyle='--', label=f'Edge-Only ({edge_num} edges)')
        
        # Plot for centralized_cloud
        cc_df = edge_num_df[edge_num_df['cluster_strategy'] == 'centralized_cloud']
        if not cc_df.empty:
            cc_df = cc_df.sort_values(by='traffic_intensity')
            plt.plot(cc_df['traffic_intensity'], cc_df[metric], marker='^', 
                     linestyle=':', label=f'Centralized Cloud')
    
    plt.xlabel('Traffic Intensity')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

# Create separate plots for each edge server number
for edge_num in edge_server_numbers:
    plt.figure(figsize=(15, 10))
    edge_num_df = filtered_df[filtered_df['edge_server_number'] == edge_num]
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        
        # Plot for edge_cloud_level_1
        ec_df = edge_num_df[edge_num_df['cluster_strategy'] == 'edge_cloud_level_1']
        if not ec_df.empty:
            ec_df = ec_df.sort_values(by='traffic_intensity')
            plt.plot(ec_df['traffic_intensity'], ec_df[metric], marker='o', 
                     label=f'Edge-Cloud')
        
        # Plot for edge_only_level_1
        eo_df = edge_num_df[edge_num_df['cluster_strategy'] == 'edge_only_level_1']
        if not eo_df.empty:
            eo_df = eo_df.sort_values(by='traffic_intensity')
            plt.plot(eo_df['traffic_intensity'], eo_df[metric], marker='s', 
                     linestyle='--', label=f'Edge-Only')
        
        # Plot for centralized_cloud
        cc_df = edge_num_df[edge_num_df['cluster_strategy'] == 'centralized_cloud']
        if not cc_df.empty:
            cc_df = cc_df.sort_values(by='traffic_intensity')
            plt.plot(cc_df['traffic_intensity'], cc_df[metric], marker='^', 
                     linestyle=':', label=f'Centralized Cloud')
        
        plt.xlabel('Traffic Intensity')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} - {edge_num} Edge Servers')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()