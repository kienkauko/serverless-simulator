import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

def load_and_process_data():
    """Load data from both Excel files representing different network levels"""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load both Excel files
    level_1_file = os.path.join(script_dir, 'population_2_live_time.xlsx')
    level_2_file = os.path.join(script_dir, 'population_2_live_time_level_2.xlsx')
    
    df_level_1 = pd.read_excel(level_1_file)
    df_level_2 = pd.read_excel(level_2_file)
    
    # Add network level labels
    df_level_1['network_level'] = 'Level 1'
    df_level_2['network_level'] = 'Level 2'
    
    # Combine both datasets
    df_combined = pd.concat([df_level_1, df_level_2], ignore_index=True)
    
    # Filter data: exclude centralized_cloud
    df_filtered = df_combined[df_combined['cluster_strategy'] != 'centralized_cloud'].copy()
    
    # For level 1 data, only keep edge_server_number = 5000
    # For level 2 data, keep all edge_server_number values (since it doesn't have different values)
    df_level_1_filtered = df_filtered[
        (df_filtered['network_level'] == 'Level 1') & 
        (df_filtered['edge_server_number'] == 5000)
    ].copy()
    
    df_level_2_filtered = df_filtered[
        (df_filtered['network_level'] == 'Level 2') & 
        (df_filtered['edge_server_number'] == 5000)
    ].copy()

    # df_level_2_filtered = df_filtered[df_filtered['network_level'] == 'Level 2'].copy()
    
    # Combine filtered data
    df_final = pd.concat([df_level_1_filtered, df_level_2_filtered], ignore_index=True)
    
    # Convert traffic_intensity to offered load percentage (0-100%)
    # traffic_intensity ranges from 0.0001 to 0.002 with step 0.0001
    # This corresponds to 0-100% offered load
    df_final['offered_load'] = ((df_final['traffic_intensity'] - 0.0001) / (0.002 - 0.0001)) * 100
    
    # Create a combined label for strategy and network level
    df_final['strategy_level'] = df_final['cluster_strategy'] + '_' + df_final['network_level'].str.replace(' ', '_')
    
    return df_final

def create_overview_comparison_plots(df):
    """Create overview comparison plots for all metrics comparing network levels"""
    
    # Define the metrics to plot
    metrics = [
        'blocking_percentage', 'avg_offloaded_to_cloud', 'avg_total_latency',
        'avg_spawn_time', 'avg_processing_time', 'avg_network_time',
        'ram_req', 'power_req'
    ]
    
    # Define better labels for metrics
    metric_labels = {
        'blocking_percentage': 'Blocking Percentage (%)',
        'avg_offloaded_to_cloud': 'Avg Offloaded to Cloud',
        'avg_total_latency': 'Avg Total Latency (ms)',
        'avg_spawn_time': 'Avg Spawn Time (ms)',
        'avg_processing_time': 'Avg Processing Time (ms)',
        'avg_network_time': 'Avg Network Time (ms)',
        'ram_req': 'RAM per request (%)',
        'power_req': 'Power per request (W)'
    }
    
    # Get unique strategy-level combinations
    strategy_levels = df['strategy_level'].unique()
    
    # Create color and style maps
    colors = ['blue', 'red', 'green', 'orange']
    color_map = {}
    linestyles = {}
    markers = {}
    
    color_idx = 0
    for combo in sorted(strategy_levels):
        if 'Level_1' in combo:
            color_map[combo] = colors[color_idx % len(colors)]
            linestyles[combo] = '-'
            markers[combo] = 'o'
        else:  # Level_2
            color_map[combo] = colors[color_idx % len(colors)]
            linestyles[combo] = '--'
            markers[combo] = 's'
        color_idx += 1
    
    # Create subplots (3x3 grid with 8 metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Network Level Comparison (Level 1 vs Level 2)\nEdge Server Placement Strategies', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        
        # Plot each strategy-level combination
        for combo in sorted(strategy_levels):
            combo_data = df[df['strategy_level'] == combo].sort_values('offered_load')
            
            # Extract strategy and level for legend
            strategy_part = combo.replace('_Level_1', '').replace('_Level_2', '')
            strategy_name = strategy_part.replace('_', ' ').title()
            
            level_part = 'Level 1' if 'Level_1' in combo else 'Level 2'
            
            label = f"{strategy_name} ({level_part})"
            
            ax.plot(combo_data['offered_load'], combo_data[metric], 
                   marker=markers[combo], linewidth=2, markersize=6, 
                   linestyle=linestyles[combo], label=label, color=color_map[combo])
        
        ax.set_xlabel('Offered Load (%)', fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_title(f'{metric_labels[metric]} vs Offered Load', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Set x-axis limits
        ax.set_xlim(0, 100)
    
    # Hide the last empty subplot
    if len(metrics) < len(axes_flat):
        axes_flat[-1].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    return fig

def create_individual_plots(df):
    """Create individual larger plots for better visibility"""
    
    metrics = [
        'blocking_percentage', 'avg_offloaded_to_cloud', 'avg_total_latency',
        'avg_spawn_time', 'avg_processing_time', 'avg_network_time',
        'ram_req', 'power_req'
    ]
    
    metric_labels = {
        'blocking_percentage': 'Blocking Percentage (%)',
        'avg_offloaded_to_cloud': 'Avg Offloaded to Cloud',
        'avg_total_latency': 'Avg Total Latency (ms)',
        'avg_spawn_time': 'Avg Spawn Time (ms)',
        'avg_processing_time': 'Avg Processing Time (ms)',
        'avg_network_time': 'Avg Network Time (ms)',
        'ram_req': 'RAM per request (%)',
        'power_req': 'Power per request (W)'
    }
    
    strategy_levels = df['strategy_level'].unique()
    
    # Create color and style maps
    colors = ['blue', 'red', 'green', 'orange']
    color_map = {}
    linestyles = {}
    markers = {}
    
    color_idx = 0
    for combo in sorted(strategy_levels):
        if 'Level_1' in combo:
            color_map[combo] = colors[color_idx % len(colors)]
            linestyles[combo] = '-'
            markers[combo] = 'o'
        else:  # Level_2
            color_map[combo] = colors[color_idx % len(colors)]
            linestyles[combo] = '--'
            markers[combo] = 's'
        color_idx += 1
    
    figures = []
    
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        for combo in sorted(strategy_levels):
            combo_data = df[df['strategy_level'] == combo].sort_values('offered_load')
            
            # Extract strategy and level for legend
            strategy_part = combo.replace('_Level_1', '').replace('_Level_2', '')
            strategy_name = strategy_part.replace('_', ' ').title()
            
            level_part = 'Level 1' if 'Level_1' in combo else 'Level 2'
            
            label = f"{strategy_name} ({level_part})"
            
            ax.plot(combo_data['offered_load'], combo_data[metric], 
                   marker=markers[combo], linewidth=3, markersize=8, 
                   linestyle=linestyles[combo], label=label, color=color_map[combo])
        
        ax.set_xlabel('Offered Load (%)', fontsize=14)
        ax.set_ylabel(metric_labels[metric], fontsize=14)
        ax.set_title(f'{metric_labels[metric]} vs Offered Load\n', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if metric == 'blocking_percentage' or metric == 'avg_total_latency':
            ax.legend(fontsize=12, loc='best')
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def create_strategy_specific_plots(df):
    """Create plots comparing network levels for each cluster strategy separately"""
    
    metrics = [
        'blocking_percentage', 'avg_offloaded_to_cloud', 'avg_total_latency',
        'avg_spawn_time', 'avg_processing_time', 'avg_network_time',
        'ram_req', 'power_req'
    ]
    
    metric_labels = {
        'blocking_percentage': 'Blocking Percentage (%)',
        'avg_offloaded_to_cloud': 'Avg Offloaded to Cloud',
        'avg_total_latency': 'Avg Total Latency (s)',
        'avg_spawn_time': 'Avg Spawn Time (s)',
        'avg_processing_time': 'Avg Processing Time (s)',
        'avg_network_time': 'Avg Network Time (s)',
        'ram_req': 'RAM per request (%)',
        'power_req': 'Power per request (W)'
    }
    
    # Get unique cluster strategies
    cluster_strategies = sorted(df['cluster_strategy'].unique())
    
    figures = []
    
    for strategy in cluster_strategies:
        # Filter data for specific cluster strategy
        df_strategy = df[df['cluster_strategy'] == strategy]
        
        if len(df_strategy) == 0:
            continue
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        strategy_title = strategy.replace('_', ' ').title()
        fig.suptitle(f'Network Level Comparison - {strategy_title}\n(Level 1 vs Level 2)', 
                     fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes_flat[i]
            
            # Plot each network level for this cluster strategy
            for level in ['Level 1', 'Level 2']:
                level_data = df_strategy[df_strategy['network_level'] == level].sort_values('offered_load')
                
                if len(level_data) > 0:
                    color = 'blue' if level == 'Level 1' else 'red'
                    marker = 'o' if level == 'Level 1' else 's'
                    linestyle = '-' if level == 'Level 1' else '--'
                    
                    ax.plot(level_data['offered_load'], level_data[metric], 
                           marker=marker, linewidth=2, markersize=6, 
                           linestyle=linestyle, label=f"{level}", 
                           color=color)
            
            ax.set_xlabel('Offered Load (%)', fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.set_title(f'{metric_labels[metric]} vs Offered Load', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 100)
        
        # Hide the last empty subplot
        if len(metrics) < len(axes_flat):
            axes_flat[-1].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        figures.append(fig)
    
    return figures

def create_performance_summary_table(df):
    """Create a summary table comparing performance metrics between network levels"""
    
    # Select key metrics for summary
    key_metrics = [
        'blocking_percentage', 'avg_total_latency', 'avg_spawn_time',
        'ram_req', 'power_req'
    ]
    
    summary_data = []
    
    for strategy in sorted(df['cluster_strategy'].unique()):
        for level in ['Level 1', 'Level 2']:
            strategy_level_data = df[
                (df['cluster_strategy'] == strategy) & 
                (df['network_level'] == level)
            ]
            
            if len(strategy_level_data) > 0:
                # Calculate average metrics across all offered loads
                row = {
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Network Level': level
                }
                
                for metric in key_metrics:
                    avg_value = strategy_level_data[metric].mean()
                    row[metric] = avg_value
                
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a figure to display the table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    headers = ['Strategy', 'Level', 'Blocking %', 'Avg Latency (ms)', 'Spawn Time (ms)', 'RAM Req (%)', 'Power Req (W)']
    
    for _, row in summary_df.iterrows():
        table_row = [
            row['Strategy'],
            row['Network Level'],
            f"{row['blocking_percentage']:.2f}",
            f"{row['avg_total_latency']:.2f}",
            f"{row['avg_spawn_time']:.3f}",
            f"{row['ram_req']:.2f}",
            f"{row['power_req']:.2f}"
        ]
        table_data.append(table_row)
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Performance Summary: Network Level Comparison\n(Average values across all offered loads)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig, summary_df

def print_data_summary(df):
    """Print summary information about the loaded data"""
    print("Network Level Comparison Analysis")
    print("=" * 50)
    print(f"Total data points: {len(df)}")
    print(f"Network levels: {sorted(df['network_level'].unique())}")
    print(f"Cluster strategies: {sorted(df['cluster_strategy'].unique())}")
    print(f"Edge server numbers (Level 1): {sorted(df[df['network_level'] == 'Level 1']['edge_server_number'].unique())}")
    print(f"Edge server numbers (Level 2): {sorted(df[df['network_level'] == 'Level 2']['edge_server_number'].unique())}")
    print(f"Traffic intensity range: {df['traffic_intensity'].min():.4f} - {df['traffic_intensity'].max():.4f}")
    print(f"Offered load range: {df['offered_load'].min():.1f}% - {df['offered_load'].max():.1f}%")
    print()
    
    # Show data distribution by strategy and level
    print("Data distribution:")
    for strategy in sorted(df['cluster_strategy'].unique()):
        for level in sorted(df['network_level'].unique()):
            count = len(df[(df['cluster_strategy'] == strategy) & (df['network_level'] == level)])
            print(f"  {strategy} - {level}: {count} data points")
    print()

def main():
    """Main function to run the analysis"""
    try:
        print("Loading data from Excel files...")
        
        # Load and process data
        df = load_and_process_data()
        
        if len(df) == 0:
            print("No data found after filtering.")
            return
        
        # Print data summary
        print_data_summary(df)
        
        # print("Creating overview comparison plot...")
        # overview_fig = create_overview_comparison_plots(df)
        
        print("Creating individual metric plots...")
        individual_figs = create_individual_plots(df)
        
        print("Creating strategy-specific comparison plots...")
        strategy_figs = create_strategy_specific_plots(df)
        
        # print("Creating performance summary table...")
        # summary_fig, summary_df = create_performance_summary_table(df)
        
        # Show all plots
        plt.show()
        
        total_figs = 1 + len(individual_figs) + len(strategy_figs) + 1
        print(f"\nPlots displayed successfully!")
        print(f"Total figures created: {total_figs}")
        print(f"- 1 overview plot")
        print(f"- {len(individual_figs)} individual metric plots")
        print(f"- {len(strategy_figs)} strategy-specific comparison plots")
        print(f"- 1 performance summary table")
        
        # Optional: Save summary to CSV
        # save_option = input("\nDo you want to save the performance summary to CSV? (y/n): ").lower()
        # if save_option == 'y':
        #     script_dir = os.path.dirname(os.path.abspath(__file__))
        #     csv_path = os.path.join(script_dir, 'network_level_comparison_summary.csv')
        #     summary_df.to_csv(csv_path, index=False)
        #     print(f"Summary saved to: {csv_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find Excel file. {str(e)}")
        print("Please make sure both 'equally_2_live_time.xlsx' and 'equally_2_live_time_level_2.xlsx' files are in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
