import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import itertools
from matplotlib.colors import to_rgba
import matplotlib.cm as cm

def main():
    # Find all files matching the pattern "*_live_time.xlsx"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    live_time_files = glob.glob(os.path.join(current_dir, "..", "*_live_time.xlsx"))
    
    if not live_time_files:
        # Try current directory if no files found in parent directory
        live_time_files = glob.glob(os.path.join(current_dir, "*_live_time.xlsx"))

    # Extract live time values from filenames and sort
    live_time_values = []
    file_paths = {}

    for file_path in live_time_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\w+)_?(\d+)_live_time\.xlsx$', filename)
        if match:
            prefix = match.group(1) if match.group(1) else ""
            live_time = int(match.group(2))
            key = (prefix, live_time)
            live_time_values.append(key)
            file_paths[key] = file_path

    # Sort live time values
    live_time_values.sort(key=lambda x: (x[0], x[1]))

    if not live_time_values:
        raise ValueError("No live time files found for analysis")

    print(f"Found live time files: {[f'{prefix}_{time}' for prefix, time in live_time_values]}")
    
    # Allow filtering specific live time values to analyze
    # If you want to analyze only specific live time values, uncomment and modify the lines below:
    # filtered_values = [(prefix, time) for prefix, time in live_time_values if time in [2, 4]]
    # live_time_values = filtered_values
    
    # Read all dataframes
    dataframes = {}
    for key in live_time_values:
        prefix, time = key
        file_path = file_paths[key]
        print(f"Reading file: {os.path.basename(file_path)}")
        df = pd.read_excel(file_path)
        # Convert traffic_intensity to percentage (offered load)
        df['offered_load'] = df['traffic_intensity'] * 100 / 0.002  # 0.002 is max value = 100%
        dataframes[key] = df

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

    # Get unique edge server numbers and strategies from all dataframes
    all_edge_server_numbers = set()
    all_strategies = set()
    
    for df in dataframes.values():
        all_edge_server_numbers.update(df['edge_server_number'].unique())
        all_strategies.update(df['cluster_strategy'].unique())
    
    edge_server_numbers = sorted(all_edge_server_numbers)
    strategies = sorted(all_strategies)
    
    print(f"Edge server numbers found: {edge_server_numbers}")
    print(f"Strategies found: {strategies}")

    # Generate colors for different live time values using a colormap
    num_live_times = len(live_time_values)
    colormap = cm.get_cmap('tab10' if num_live_times <= 10 else 'tab20', max(10, num_live_times))
    
    colors = {}
    for i, key in enumerate(live_time_values):
        prefix, time = key
        label = f"{prefix}_{time}" if prefix else f"{time}"
        colors[label] = colormap(i % colormap.N)
    
    # Markers for different strategies
    markers = {
        'massive_edge_cloud': 'o', 
        'massive_edge': 's',
        'edge_cloud_level_1': '^', 
        'edge_only_level_1': 'x',
        'centralized_cloud': 'd'
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
    
    for key in live_time_values:
        prefix, time = key
        label = f"{prefix}_{time}" if prefix else f"{time}"
        df = dataframes[key]
        
        plt.figure(figsize=(26, 16))
        plt.suptitle(f"Live Time {label} Metrics", fontsize=16)
        
        df_edge_numbers = sorted(df['edge_server_number'].unique())
        df_strategies = sorted(df['cluster_strategy'].unique())

        for i, metric in enumerate(metrics, 1):
            plt.subplot(3, 4, i)
            
            for edge_num in df_edge_numbers:
                # Filter data for the current edge server number
                edge_num_df = df[df['edge_server_number'] == edge_num]
                
                for strategy in df_strategies:
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

    # Part 2: Overview comparison - show all live times for each strategy/edge combo
    # =============================================================================
    
    # For each strategy and edge server number combination
    for strategy in strategies:
        for edge_num in edge_server_numbers:
            # Check if this combination exists in at least one dataframe
            valid_data = False
            for key in live_time_values:
                df = dataframes[key]
                if not df[(df['cluster_strategy'] == strategy) & (df['edge_server_number'] == edge_num)].empty:
                    valid_data = True
                    break
            
            if not valid_data:
                continue
                
            plt.figure(figsize=(26, 16))
            plt.suptitle(f"Live Time Comparison: {strategy} with {edge_num} Edge Servers", fontsize=16)
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(3, 4, i)
                
                for key in live_time_values:
                    prefix, time = key
                    label = f"{prefix}_{time}" if prefix else f"{time}"
                    df = dataframes[key]
                    
                    # Filter data for the specific strategy and edge number
                    data = df[(df['cluster_strategy'] == strategy) & (df['edge_server_number'] == edge_num)]
                    
                    if not data.empty:
                        data = data.sort_values(by='offered_load')
                        plt.plot(
                            data['offered_load'], 
                            data[metric],
                            color=colors[label],
                            marker='o',
                            linestyle='-',
                            label=f'Live Time {label}'
                        )
                
                plt.xlabel('Offered Load (%)')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.title(f'{metric.replace("_", " ").title()}')
                plt.grid(True)
                if i == 1:  # Only add legend to the first subplot to save space
                    plt.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2, w_pad=1.0, h_pad=0.8)
            plt.show()
    
    # Part 3: Pairwise comparisons (if there are at least 2 live time files)
    # ====================================================================
    
    if len(live_time_values) >= 2:
        print("Performing pairwise comparisons between live time configurations...")
        
        # Generate all possible pairs of live time values
        pairs = list(itertools.combinations(live_time_values, 2))
        
        for pair in pairs:
            key1, key2 = pair
            prefix1, time1 = key1
            prefix2, time2 = key2
            label1 = f"{prefix1}_{time1}" if prefix1 else f"{time1}"
            label2 = f"{prefix2}_{time2}" if prefix2 else f"{time2}"
            
            print(f"Comparing Live Time {label1} vs Live Time {label2}")
            
            df1 = dataframes[key1]
            df2 = dataframes[key2]
            
            # Find common strategies and edge numbers
            common_strategies = set(df1['cluster_strategy'].unique()).intersection(
                set(df2['cluster_strategy'].unique()))
            common_edge_numbers = set(df1['edge_server_number'].unique()).intersection(
                set(df2['edge_server_number'].unique()))
            
            print(f"Common strategies: {common_strategies}")
            print(f"Common edge server numbers: {common_edge_numbers}")
            
            # For each strategy and edge server number combination, plot direct comparison
            for strategy in common_strategies:
                for edge_num in common_edge_numbers:
                    # Filter data for the specific strategy and edge number
                    data1 = df1[(df1['cluster_strategy'] == strategy) & 
                               (df1['edge_server_number'] == edge_num)]
                    data2 = df2[(df2['cluster_strategy'] == strategy) & 
                               (df2['edge_server_number'] == edge_num)]
                    
                    if data1.empty or data2.empty:
                        print(f"Missing data for {strategy} with {edge_num} edge servers in one of the live time configurations")
                        continue
                    
                    # Sort by offered load
                    data1 = data1.sort_values(by='offered_load')
                    data2 = data2.sort_values(by='offered_load')
                    
                    # Direct comparison plots
                    plt.figure(figsize=(26, 16))
                    plt.suptitle(f"Direct Comparison: {strategy} with {edge_num} Edge Servers - {label1} vs {label2}", fontsize=16)
                    
                    for i, metric in enumerate(metrics, 1):
                        plt.subplot(3, 4, i)
                        
                        plt.plot(
                            data1['offered_load'], 
                            data1[metric],
                            color=colors[label1],
                            marker='o',
                            linestyle='-',
                            label=f'Live Time {label1}'
                        )
                        
                        plt.plot(
                            data2['offered_load'], 
                            data2[metric],
                            color=colors[label2],
                            marker='s',
                            linestyle='--',
                            label=f'Live Time {label2}'
                        )
                        
                        plt.xlabel('Offered Load (%)')
                        plt.ylabel(metric.replace('_', ' ').title())
                        plt.title(f'{metric.replace("_", " ").title()}')
                        plt.grid(True)
                        if i == 1:
                            plt.legend()
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2, w_pad=1.0, h_pad=0.8)
                    plt.show()
                    
                    # Percentage difference plots
                    plt.figure(figsize=(26, 16))
                    plt.suptitle(f"Percentage Difference: Live Time {label2} vs Live Time {label1} ({strategy}, {edge_num} edges)", fontsize=16)
                    
                    # Find common offered loads
                    common_loads = set(data1['offered_load']).intersection(set(data2['offered_load']))
                    if not common_loads:
                        print(f"No common offered loads for {strategy} with {edge_num} edges")
                        continue
                    
                    # Filter to common loads
                    filtered_data1 = data1[data1['offered_load'].isin(common_loads)]
                    filtered_data2 = data2[data2['offered_load'].isin(common_loads)]
                    
                    for i, metric in enumerate(metrics, 1):
                        plt.subplot(3, 4, i)
                        
                        # Ensure data is aligned by offered load
                        diff_data = []
                        loads = []
                        
                        for load in sorted(common_loads):
                            val1 = filtered_data1[filtered_data1['offered_load'] == load][metric].values
                            val2 = filtered_data2[filtered_data2['offered_load'] == load][metric].values
                            
                            if len(val1) > 0 and len(val2) > 0:
                                # Calculate percentage difference: (live_time2 - live_time1) / live_time1 * 100
                                # Handle division by zero
                                if val1[0] != 0:
                                    pct_diff = (val2[0] - val1[0]) / val1[0] * 100
                                else:
                                    if val2[0] == 0:
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
