import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

def load_and_process_data():
    """Load data from Excel file and process it according to requirements"""
    # Load the Excel file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equally_2_live_time.xlsx')
    df = pd.read_excel(file_path)
    
    # Filter data: exclude centralized_cloud, only keep massive_edge_cloud and massive_edge
    df_filtered = df[df['cluster_strategy'].isin(['massive_edge_cloud', 'massive_edge'])].copy()
    
    # Convert traffic_intensity to offered load percentage (0-100%)
    # traffic_intensity ranges from 0.0001 to 0.002 with step 0.0001
    # This corresponds to 0-100% offered load
    df_filtered['offered_load'] = ((df_filtered['traffic_intensity'] - 0.0001) / (0.002 - 0.0001)) * 100
    
    # Create a combined label for strategy and edge server number
    df_filtered['strategy_servers'] = df_filtered['cluster_strategy'] + '_' + df_filtered['edge_server_number'].astype(str)
    
    return df_filtered

def create_comparison_plots(df):
    """Create comparison plots for different metrics comparing strategies with different edge server numbers"""
    
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
    
    # Get unique strategy-server combinations
    strategy_servers = df['strategy_servers'].unique()
    
    # Create color and style maps
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategy_servers)))
    color_map = dict(zip(strategy_servers, colors))
    
    # Define line styles for different strategies
    linestyles = {}
    markers = {}
    for combo in strategy_servers:
        if 'massive_edge_cloud' in combo:
            linestyles[combo] = '-'
            markers[combo] = 'o'
        else:  # massive_edge
            linestyles[combo] = '--'
            markers[combo] = 's'
    
    # Create subplots (3x3 grid with 8 metrics)
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Edge Server Number Comparison: Massive Edge Cloud vs Massive Edge', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        
        # Plot each strategy-server combination
        for combo in sorted(strategy_servers):
            combo_data = df[df['strategy_servers'] == combo].sort_values('offered_load')
            
            # Extract strategy and server number for legend
            if 'massive_edge_cloud' in combo:
                strategy_name = "Massive Edge Cloud"
            else:  # massive_edge
                strategy_name = "Massive Edge"
            
            server_num = combo.split('_')[-1]
            label = f"{strategy_name} ({server_num} servers)"
            
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
    plt.subplots_adjust(top=0.93)
    
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
    
    strategy_servers = df['strategy_servers'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategy_servers)))
    color_map = dict(zip(strategy_servers, colors))
    
    # Define line styles and markers for different strategies
    linestyles = {}
    markers = {}
    for combo in strategy_servers:
        if 'massive_edge_cloud' in combo:
            linestyles[combo] = '-'
            markers[combo] = 'o'
        else:  # massive_edge
            linestyles[combo] = '--'
            markers[combo] = 's'
    
    figures = []
    
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        for combo in sorted(strategy_servers):
            combo_data = df[df['strategy_servers'] == combo].sort_values('offered_load')
            
            # Extract strategy and server number for legend
            if 'massive_edge_cloud' in combo:
                strategy_name = "Massive Edge Cloud"
            else:  # massive_edge
                strategy_name = "Massive Edge"
            
            server_num = combo.split('_')[-1]
            label = f"{strategy_name} ({server_num} servers)"
            
            ax.plot(combo_data['offered_load'], combo_data[metric], 
                   marker=markers[combo], linewidth=3, markersize=8, 
                   linestyle=linestyles[combo], label=label, color=color_map[combo])
        
        ax.set_xlabel('Offered Load (%)', fontsize=14)
        ax.set_ylabel(metric_labels[metric], fontsize=14)
        ax.set_title(f'{metric_labels[metric]} vs Offered Load\n', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for blocking percentage plot
        if metric == 'blocking_percentage' or metric =='avg_total_latency':
            ax.legend(fontsize=12, loc='best')
        
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def create_strategy_comparison_plots(df):
    """Create plots comparing strategies for each edge server number separately"""
    
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
    
    # Get unique edge server numbers
    edge_numbers = sorted(df['edge_server_number'].unique())
    
    figures = []
    
    for edge_num in edge_numbers:
        # Filter data for specific edge server number
        df_edge = df[df['edge_server_number'] == edge_num]
        
        if len(df_edge) == 0:
            continue
            
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Strategy Comparison with {edge_num} Edge Servers', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes_flat[i]
            
            # Plot each strategy for this edge server number
            for strategy in ['massive_edge_cloud', 'massive_edge']:
                strategy_data = df_edge[df_edge['cluster_strategy'] == strategy].sort_values('offered_load')
                
                if len(strategy_data) > 0:
                    color = 'blue' if strategy == 'massive_edge_cloud' else 'red'
                    marker = 'o' if strategy == 'massive_edge_cloud' else 's'
                    linestyle = '-' if strategy == 'massive_edge_cloud' else '--'
                    
                    ax.plot(strategy_data['offered_load'], strategy_data[metric], 
                           marker=marker, linewidth=2, markersize=6, 
                           linestyle=linestyle, label=strategy.replace('_', ' ').title(), 
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
        plt.subplots_adjust(top=0.93)
        figures.append(fig)
    
    return figures

def main():
    """Main function to run the analysis"""
    try:
        # Load and process data
        print("Loading data from equally_2_live_time.xlsx...")
        df = load_and_process_data()
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Cluster strategies found: {df['cluster_strategy'].unique()}")
        print(f"Edge server numbers found: {sorted(df['edge_server_number'].unique())}")
        print(f"Strategy-server combinations: {sorted(df['strategy_servers'].unique())}")
        print(f"Offered load range: {df['offered_load'].min():.1f}% - {df['offered_load'].max():.1f}%")
        
        # Create overview plot comparing all strategy-server combinations
        print("\nCreating overview comparison plot...")
        overview_fig = create_comparison_plots(df)
        
        # Create individual plots for better visibility
        print("Creating individual metric plots...")
        individual_figs = create_individual_plots(df)
        
        # Create strategy comparison plots for each edge server number
        print("Creating strategy comparison plots for each edge server number...")
        strategy_figs = create_strategy_comparison_plots(df)
        
        # Show all plots
        plt.show()
        
        total_figs = 1 + len(individual_figs) + len(strategy_figs)
        print(f"\nPlots displayed successfully!")
        print(f"Total figures created: {total_figs}")
        print(f"- 1 overview plot")
        print(f"- {len(individual_figs)} individual metric plots")
        print(f"- {len(strategy_figs)} strategy comparison plots")
        
    except FileNotFoundError:
        print("Error: Could not find 'equally_2_live_time.xlsx' file.")
        print("Please make sure the file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
