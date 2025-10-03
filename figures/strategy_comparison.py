import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the data from Excel files
file_path1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equally_2_live_time.xlsx')
file_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'population_2_live_time.xlsx')

df_equally = pd.read_excel(file_path1)
df_population = pd.read_excel(file_path2)

print("Equally distribution strategy data shape:", df_equally.shape)
print("Population distribution strategy data shape:", df_population.shape)

# Skip centralized_cloud scenario in both dataframes
df_equally = df_equally[df_equally['cluster_strategy'] != 'centralized_cloud']
df_population = df_population[df_population['cluster_strategy'] != 'centralized_cloud']

# Convert traffic_intensity to percentage (offered load)
df_equally['offered_load'] = df_equally['traffic_intensity'] * 100 / 0.002  # 0.002 is max value = 100%
df_population['offered_load'] = df_population['traffic_intensity'] * 100 / 0.002

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
edge_server_numbers_equally = sorted(df_equally['edge_server_number'].unique())
edge_server_numbers_population = sorted(df_population['edge_server_number'].unique())
strategies_equally = sorted(df_equally['cluster_strategy'].unique())
strategies_population = sorted(df_population['cluster_strategy'].unique())

print(f"Edge server numbers in equally distribution: {edge_server_numbers_equally}")
print(f"Edge server numbers in population distribution: {edge_server_numbers_population}")
print(f"Strategies in equally distribution: {strategies_equally}")
print(f"Strategies in population distribution: {strategies_population}")

# Set up colors and markers for different strategies and distribution methods
colors = {'equally': 'blue', 'population': 'red'}
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

# Part 1: Plot metrics for each distribution method separately
# =========================================================

# For equally distribution strategy
plt.figure(figsize=(22, 16))
plt.suptitle("Equally Distribution Strategy Metrics", fontsize=16)

for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 4, i)
    
    for edge_num in edge_server_numbers_equally:
        # Filter data for the current edge server number
        edge_num_df = df_equally[df_equally['edge_server_number'] == edge_num]
        
        for strategy in strategies_equally:
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
    else:
        plt.tight_layout()

plt.tight_layout(rect=[0, 0, 0, 0])
plt.show()

# For population distribution strategy
plt.figure(figsize=(22, 16))
plt.suptitle("Population Distribution Strategy Metrics", fontsize=16)

for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 4, i)
    
    for edge_num in edge_server_numbers_population:
        # Filter data for the current edge server number
        edge_num_df = df_population[df_population['edge_server_number'] == edge_num]
        
        for strategy in strategies_population:
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
    else:
        plt.tight_layout()

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()

# Part 2: Compare equally vs population for each metric and strategy
# ================================================================

# Determine common strategies and edge server numbers for direct comparison
common_strategies = set(strategies_equally).intersection(set(strategies_population))
common_edge_numbers = set(edge_server_numbers_equally).intersection(set(edge_server_numbers_population))

print(f"Common strategies: {common_strategies}")
print(f"Common edge server numbers: {common_edge_numbers}")

# For each strategy and edge server number combination, plot comparison
for strategy in common_strategies:
    for edge_num in common_edge_numbers:
        plt.figure(figsize=(22, 16))
        plt.suptitle(f"Comparison: {strategy} with {edge_num} Edge Servers", fontsize=16)
        
        # Filter data for the specific strategy and edge number
        equally_data = df_equally[(df_equally['cluster_strategy'] == strategy) & 
                                (df_equally['edge_server_number'] == edge_num)]
        population_data = df_population[(df_population['cluster_strategy'] == strategy) & 
                                      (df_population['edge_server_number'] == edge_num)]
        
        if equally_data.empty or population_data.empty:
            print(f"Missing data for {strategy} with {edge_num} edge servers in one of the distributions")
            continue
            
        # Sort by offered load
        equally_data = equally_data.sort_values(by='offered_load')
        population_data = population_data.sort_values(by='offered_load')
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(3, 4, i)
            
            plt.plot(equally_data['offered_load'], equally_data[metric], 
                    color=colors['equally'], marker='o', linestyle='-', 
                    label='Equally Distribution')
            
            plt.plot(population_data['offered_load'], population_data[metric], 
                    color=colors['population'], marker='s', linestyle='--', 
                    label='Population Distribution')
            
            plt.xlabel('Offered Load (%)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.grid(True)
            if i == 1:  # Only add legend to the first subplot to save space
                plt.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Part 3: Plot percentage differences between strategies
# ===================================================

print("Calculating percentage differences between distribution strategies...")

for strategy in common_strategies:
    for edge_num in common_edge_numbers:
        plt.figure(figsize=(22, 16))
        plt.suptitle(f"Percentage Difference: Population vs Equally ({strategy}, {edge_num} edges)", fontsize=16)
        
        # Filter and sort data
        equally_data = df_equally[(df_equally['cluster_strategy'] == strategy) & 
                                (df_equally['edge_server_number'] == edge_num)].sort_values(by='offered_load')
        population_data = df_population[(df_population['cluster_strategy'] == strategy) & 
                                      (df_population['edge_server_number'] == edge_num)].sort_values(by='offered_load')
        
        if equally_data.empty or population_data.empty:
            continue
            
        # Find common offered loads
        common_loads = set(equally_data['offered_load']).intersection(set(population_data['offered_load']))
        if not common_loads:
            print(f"No common offered loads for {strategy} with {edge_num} edges")
            continue
            
        # Filter to common loads
        equally_data = equally_data[equally_data['offered_load'].isin(common_loads)]
        population_data = population_data[population_data['offered_load'].isin(common_loads)]
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(3, 4, i)
            
            # Ensure data is aligned by offered load
            diff_data = []
            loads = []
            
            for load in sorted(common_loads):
                e_val = equally_data[equally_data['offered_load'] == load][metric].values
                p_val = population_data[population_data['offered_load'] == load][metric].values
                
                if len(e_val) > 0 and len(p_val) > 0:
                    # Calculate percentage difference: (population - equally) / equally * 100
                    # Handle division by zero
                    if e_val[0] != 0:
                        pct_diff = (p_val[0] - e_val[0]) / e_val[0] * 100
                    else:
                        if p_val[0] == 0:
                            pct_diff = 0  # Both are zero
                        else:
                            pct_diff = 100  # Arbitrarily set to 100% when equally is zero but population is not
                    
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
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

print("All plots have been generated.")
