import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the data from Excel files
file_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equally_2_live_time.xlsx')
file_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equally_4_live_time.xlsx')

df_live_2 = pd.read_excel(file_path1)
df_live_4 = pd.read_excel(file_path2)

print("Live time 2 data shape:", df_live_2.shape)
print("Live time 4 data shape:", df_live_4.shape)

# Keep all scenarios including centralized_cloud (no filtering needed)
# df_live_0 and df_live_2 already contain all scenarios

# Convert traffic_intensity to percentage (offered load)
df_live_2['offered_load'] = df_live_2['traffic_intensity'] * 100 / 0.002  # 0.002 is max value = 100%
df_live_4['offered_load'] = df_live_4['traffic_intensity'] * 100 / 0.002

# Define metrics to plot
metrics = [
    'blocking_percentage',
    'avg_offloaded_to_cloud',
    'avg_total_latency',
    'avg_spawn_time',
    'avg_processing_time',
    'avg_network_time',
    'mean_power',
    'mean_ram',
    'mean_cpu',
    'ram_req',
    'power_req'
]

# Get unique edge server numbers and strategies from both dataframes
edge_server_numbers_live_2 = sorted(df_live_2['edge_server_number'].unique())
edge_server_numbers_live_4 = sorted(df_live_4['edge_server_number'].unique())
strategies_live_2 = sorted(df_live_2['cluster_strategy'].unique())
strategies_live_4 = sorted(df_live_4['cluster_strategy'].unique())

print(f"Edge server numbers in live time 2: {edge_server_numbers_live_2}")
print(f"Edge server numbers in live time 4: {edge_server_numbers_live_4}")
print(f"Strategies in live time 2: {strategies_live_2}")
print(f"Strategies in live time 4: {strategies_live_4}")

# Set up colors and markers for different strategies and live time configurations
colors = {'live_2': 'blue', 'live_4': 'red'}
markers = {
    'massive_edge_cloud': 'o', 
    'massive_edge': 's',
    'edge_cloud_level_1': '^', 
    'edge_only_level_1': 'x'
}
linestyles = {
    'massive_edge_cloud': '-', 
    'massive_edge': '--',
    'edge_cloud_level_1': '-.', 
    'edge_only_level_1': ':'
}

# Part 1: Plot metrics for each live time separately
# =========================================================

# For Live time 2
plt.figure(figsize=(26, 16))
plt.suptitle("Live Time 2 Metrics", fontsize=16)

for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 4, i)
    
    for edge_num in edge_server_numbers_live_2:
        # Filter data for the current edge server number
        edge_num_df = df_live_2[df_live_2['edge_server_number'] == edge_num]
        
        for strategy in strategies_live_2:
            # Filter data for the current strategy
            strat_df = edge_num_df[edge_num_df['cluster_strategy'] == strategy]
            
            if not strat_df.empty:
                strat_df = strat_df.sort_values(by='offered_load')
                plt.plot(
                    strat_df['offered_load'], 
                    strat_df[metric], 
                    marker=markers.get(strategy, 'o'),
                    linestyle=linestyles.get(strategy, '-'),
                    label=f'{strategy} ({edge_num} edges)'
                )
    
    plt.xlabel('Offered Load (%)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()}')
    plt.grid(True)
    if i == 1:  # Only add legend to the first subplot to save space
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.95, 0.95], pad=1.2, w_pad=1.0, h_pad=0.8)
plt.show()

# For Live time 4
plt.figure(figsize=(26, 16))
plt.suptitle("Live Time 4 Metrics", fontsize=16)

for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 4, i)
    
    for edge_num in edge_server_numbers_live_2:
        # Filter data for the current edge server number
        edge_num_df = df_live_2[df_live_2['edge_server_number'] == edge_num]
        
        for strategy in strategies_live_2:
            # Filter data for the current strategy
            strat_df = edge_num_df[edge_num_df['cluster_strategy'] == strategy]
            
            if not strat_df.empty:
                strat_df = strat_df.sort_values(by='offered_load')
                plt.plot(
                    strat_df['offered_load'], 
                    strat_df[metric], 
                    marker=markers.get(strategy, 'o'),
                    linestyle=linestyles.get(strategy, '-'),
                    label=f'{strategy} ({edge_num} edges)'
                )
    
    plt.xlabel('Offered Load (%)')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()}')
    plt.grid(True)
    if i == 1:  # Only add legend to the first subplot to save space
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.95, 0.95], pad=1.2, w_pad=1.0, h_pad=0.8)
plt.show()

# Part 2: Compare Live Time 0 vs Live Time 2 for each metric, strategy, and edge server number
# =========================================================================================

# Determine common strategies and edge server numbers for direct comparison
common_strategies = set(strategies_live_2).intersection(set(strategies_live_4))
common_edge_numbers = set(edge_server_numbers_live_2).intersection(set(edge_server_numbers_live_4))

print(f"Common strategies: {common_strategies}")
print(f"Common edge server numbers: {common_edge_numbers}")

# For each strategy and edge server number combination, plot comparison
for strategy in common_strategies:
    for edge_num in common_edge_numbers:
        plt.figure(figsize=(26, 16))
        plt.suptitle(f"Comparison: {strategy} with {edge_num} Edge Servers - Live Time Impact", fontsize=16)
        
        # Filter data for the specific strategy and edge number
        live_2_data = df_live_2[(df_live_2['cluster_strategy'] == strategy) & 
                              (df_live_2['edge_server_number'] == edge_num)]
        live_4_data = df_live_4[(df_live_4['cluster_strategy'] == strategy) & 
                              (df_live_4['edge_server_number'] == edge_num)]
        
        if live_2_data.empty or live_4_data.empty:
            print(f"Missing data for {strategy} with {edge_num} edge servers in one of the live time configurations")
            continue
            
        # Sort by offered load
        live_2_data = live_2_data.sort_values(by='offered_load')
        live_4_data = live_4_data.sort_values(by='offered_load')
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(3, 4, i)
            
            plt.plot(live_2_data['offered_load'], live_2_data[metric], 
                   color=colors['live_2'], marker='o', linestyle='-', 
                   label='Live Time 2')
            
            plt.plot(live_4_data['offered_load'], live_4_data[metric], 
                   color=colors['live_4'], marker='s', linestyle='--', 
                   label='Live Time 4')
            
            plt.xlabel('Offered Load (%)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.grid(True)
            if i == 1:  # Only add legend to the first subplot to save space
                plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2, w_pad=1.0, h_pad=0.8)
        plt.show()

# Part 3: Plot percentage differences between live time configurations
# ===================================================================

print("Calculating percentage differences between live time configurations...")

for strategy in common_strategies:
    for edge_num in common_edge_numbers:
        plt.figure(figsize=(26, 16))
        plt.suptitle(f"Percentage Difference: Live Time 4 vs Live Time 2 ({strategy}, {edge_num} edges)", fontsize=16)
        
        # Filter and sort data
        live_2_data = df_live_2[(df_live_2['cluster_strategy'] == strategy) & 
                              (df_live_2['edge_server_number'] == edge_num)].sort_values(by='offered_load')
        live_4_data = df_live_4[(df_live_4['cluster_strategy'] == strategy) & 
                              (df_live_4['edge_server_number'] == edge_num)].sort_values(by='offered_load')
        
        if live_2_data.empty or live_4_data.empty:
            continue
            
        # Find common offered loads
        common_loads = set(live_2_data['offered_load']).intersection(set(live_4_data['offered_load']))
        if not common_loads:
            print(f"No common offered loads for {strategy} with {edge_num} edges")
            continue
            
        # Filter to common loads
        live_2_data = live_2_data[live_2_data['offered_load'].isin(common_loads)]
        live_4_data = live_4_data[live_4_data['offered_load'].isin(common_loads)]
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(3, 4, i)
            
            # Ensure data is aligned by offered load
            diff_data = []
            loads = []
            
            for load in sorted(common_loads):
                live_2_val = live_2_data[live_2_data['offered_load'] == load][metric].values
                live_4_val = live_4_data[live_4_data['offered_load'] == load][metric].values
                
                if len(live_2_val) > 0 and len(live_4_val) > 0:
                    # Calculate percentage difference: (live_4 - live_2) / live_2 * 100
                    # Handle division by zero
                    if live_2_val[0] != 0:
                        pct_diff = (live_4_val[0] - live_2_val[0]) / live_2_val[0] * 100
                    else:
                        if live_4_val[0] == 0:
                            pct_diff = 0  # Both are zero
                        else:
                            pct_diff = 100  # Arbitrarily set to 100% when live_2 is zero but live_4 is not
                    
                    diff_data.append(pct_diff)
                    loads.append(load)
            
            bars = plt.bar(loads, diff_data, width=2.0)  # Increased width for better visibility
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('Offered Load (%)')
            plt.ylabel('Percentage Difference (%)')
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            
            # Only annotate the highest and lowest values
            if diff_data:
                max_idx = np.argmax(diff_data)
                min_idx = np.argmin(diff_data)
                
                # Annotate max value
                plt.text(loads[max_idx], diff_data[max_idx], 
                         f"{diff_data[max_idx]:.1f}%", 
                         ha='center', 
                         va='bottom' if diff_data[max_idx] >= 0 else 'top',
                         fontweight='bold')
                
                # Annotate min value (if different from max)
                if max_idx != min_idx:
                    plt.text(loads[min_idx], diff_data[min_idx], 
                             f"{diff_data[min_idx]:.1f}%", 
                             ha='center', 
                             va='bottom' if diff_data[min_idx] >= 0 else 'top',
                             fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2, w_pad=1.0, h_pad=0.8)
        plt.show()

print("All plots have been generated.")
