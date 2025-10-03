import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

def main():
    # Find all files matching the pattern "*_live_time.xlsx"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    live_time_files = glob.glob(os.path.join(current_dir, "*_live_time.xlsx"))

    # Extract live time values from filenames and sort
    live_time_values = []
    file_paths = {}

    for file_path in live_time_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\d+)_live_time\.xlsx$', filename)
        if match:
            live_time = int(match.group(1))
            live_time_values.append(live_time)
            file_paths[live_time] = file_path

    # Sort live time values
    live_time_values.sort()

    if len(live_time_values) < 1:
        raise ValueError("No live time files found for analysis")

    print(f"Found live time files with values: {live_time_values}")
    
    # Allow filtering specific live time values to analyze
    # If you want to analyze only specific live time values, uncomment and modify the line below:
    # live_time_values = [2, 4]  # Example: analyze only live time 2 and 4
    
    # Allow specifying specific live time values to compare
    # live_time1 = 2  # Uncomment and specify values if needed
    # live_time2 = 4  # Uncomment and specify values if needed

    # Read the data from Excel files
    file_path1 = file_paths[live_time1]
    file_path2 = file_paths[live_time2]

    print(f"Comparing live time {live_time1} and {live_time2}")
    print(f"File 1: {os.path.basename(file_path1)}")
    print(f"File 2: {os.path.basename(file_path2)}")

    # Read the dataframes
    df_live_time1 = pd.read_excel(file_path1)
    df_live_time2 = pd.read_excel(file_path2)

    print(f"Live time {live_time1} data shape:", df_live_time1.shape)
    print(f"Live time {live_time2} data shape:", df_live_time2.shape)

    # Keep all scenarios including centralized_cloud (no filtering needed)
    # Both dataframes already contain all scenarios

    # Convert traffic_intensity to percentage (offered load)
    df_live_time1['offered_load'] = df_live_time1['traffic_intensity'] * 100 / 0.002  # 0.002 is max value = 100%
    df_live_time2['offered_load'] = df_live_time2['traffic_intensity'] * 100 / 0.002

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
    edge_server_numbers1 = sorted(df_live_time1['edge_server_number'].unique())
    edge_server_numbers2 = sorted(df_live_time2['edge_server_number'].unique())
    strategies1 = sorted(df_live_time1['cluster_strategy'].unique())
    strategies2 = sorted(df_live_time2['cluster_strategy'].unique())

    print(f"Edge server numbers in live time {live_time1}: {edge_server_numbers1}")
    print(f"Edge server numbers in live time {live_time2}: {edge_server_numbers2}")
    print(f"Strategies in live time {live_time1}: {strategies1}")
    print(f"Strategies in live time {live_time2}: {strategies2}")

    # Set up colors and markers for different strategies and live time configurations
    colors = {f'live_{live_time1}': 'blue', f'live_{live_time2}': 'red'}
    markers = {
        'massive_edge_cloud': 'o', 
        'massive_edge': 's',
        'edge_cloud_level_1': '^', 
        'edge_only_level_1': 'x',
        'centralized_cloud': 'd'  # Diamond marker for centralized cloud
    }
    linestyles = {
        'massive_edge_cloud': '-', 
        'massive_edge': '--',
        'edge_cloud_level_1': '-.', 
        'edge_only_level_1': ':',
        'centralized_cloud': '-.'
    }

    # Part 1: Plot metrics for each live time separately
    # =========================================================

    # For first live time
    plt.figure(figsize=(26, 16))
    plt.suptitle(f"Live Time {live_time1} Metrics", fontsize=16)

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 4, i)
        
        for edge_num in edge_server_numbers1:
            # Filter data for the current edge server number
            edge_num_df = df_live_time1[df_live_time1['edge_server_number'] == edge_num]
            
            for strategy in strategies1:
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

    # For second live time
    plt.figure(figsize=(26, 16))
    plt.suptitle(f"Live Time {live_time2} Metrics", fontsize=16)

    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 4, i)
        
        for edge_num in edge_server_numbers2:
            # Filter data for the current edge server number
            edge_num_df = df_live_time2[df_live_time2['edge_server_number'] == edge_num]
            
            for strategy in strategies2:
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

    # Part 2: Compare Live Times for each metric, strategy, and edge server number
    # =========================================================================================

    # Determine common strategies and edge server numbers for direct comparison
    common_strategies = set(strategies1).intersection(set(strategies2))
    common_edge_numbers = set(edge_server_numbers1).intersection(set(edge_server_numbers2))

    print(f"Common strategies: {common_strategies}")
    print(f"Common edge server numbers: {common_edge_numbers}")

    # For each strategy and edge server number combination, plot comparison
    for strategy in common_strategies:
        for edge_num in common_edge_numbers:
            plt.figure(figsize=(26, 16))
            plt.suptitle(f"Comparison: {strategy} with {edge_num} Edge Servers - Live Time Impact", fontsize=16)
            
            # Filter data for the specific strategy and edge number
            live_time1_data = df_live_time1[(df_live_time1['cluster_strategy'] == strategy) & 
                                      (df_live_time1['edge_server_number'] == edge_num)]
            live_time2_data = df_live_time2[(df_live_time2['cluster_strategy'] == strategy) & 
                                      (df_live_time2['edge_server_number'] == edge_num)]
            
            if live_time1_data.empty or live_time2_data.empty:
                print(f"Missing data for {strategy} with {edge_num} edge servers in one of the live time configurations")
                continue
                
            # Sort by offered load
            live_time1_data = live_time1_data.sort_values(by='offered_load')
            live_time2_data = live_time2_data.sort_values(by='offered_load')
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(3, 4, i)
                
                plt.plot(live_time1_data['offered_load'], live_time1_data[metric], 
                       color=colors[f'live_{live_time1}'], marker='o', linestyle='-', 
                       label=f'Live Time {live_time1}')
                
                plt.plot(live_time2_data['offered_load'], live_time2_data[metric], 
                       color=colors[f'live_{live_time2}'], marker='s', linestyle='--', 
                       label=f'Live Time {live_time2}')
                
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
            plt.suptitle(f"Percentage Difference: Live Time {live_time2} vs Live Time {live_time1} ({strategy}, {edge_num} edges)", fontsize=16)
            
            # Filter and sort data
            live_time1_data = df_live_time1[(df_live_time1['cluster_strategy'] == strategy) & 
                                      (df_live_time1['edge_server_number'] == edge_num)].sort_values(by='offered_load')
            live_time2_data = df_live_time2[(df_live_time2['cluster_strategy'] == strategy) & 
                                      (df_live_time2['edge_server_number'] == edge_num)].sort_values(by='offered_load')
            
            if live_time1_data.empty or live_time2_data.empty:
                continue
                
            # Find common offered loads
            common_loads = set(live_time1_data['offered_load']).intersection(set(live_time2_data['offered_load']))
            if not common_loads:
                print(f"No common offered loads for {strategy} with {edge_num} edges")
                continue
                
            # Filter to common loads
            live_time1_data = live_time1_data[live_time1_data['offered_load'].isin(common_loads)]
            live_time2_data = live_time2_data[live_time2_data['offered_load'].isin(common_loads)]
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(3, 4, i)
                
                # Ensure data is aligned by offered load
                diff_data = []
                loads = []
                
                for load in sorted(common_loads):
                    live1_val = live_time1_data[live_time1_data['offered_load'] == load][metric].values
                    live2_val = live_time2_data[live_time2_data['offered_load'] == load][metric].values
                    
                    if len(live1_val) > 0 and len(live2_val) > 0:
                        # Calculate percentage difference: (live_time2 - live_time1) / live_time1 * 100
                        # Handle division by zero
                        if live1_val[0] != 0:
                            pct_diff = (live2_val[0] - live1_val[0]) / live1_val[0] * 100
                        else:
                            if live2_val[0] == 0:
                                pct_diff = 0  # Both are zero
                            else:
                                pct_diff = 100  # Arbitrarily set to 100% when live_time1 is zero but live_time2 is not
                        
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

if __name__ == "__main__":
    main()
