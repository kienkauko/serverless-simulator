import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

def load_and_process_data():
    """Load data from Excel file and process it according to requirements"""
    # Load the Excel file
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equally_2_live_time.xlsx')

    df = pd.read_excel(file_path)
    
    # Filter data: only keep edge_server_number = 5000 for massive_edge_cloud and massive_edge
    df_filtered = df.copy()
    massive_strategies = ['massive_edge_cloud', 'massive_edge']
    mask = (df_filtered['cluster_strategy'].isin(massive_strategies)) & (df_filtered['edge_server_number'] != 5000)
    df_filtered = df_filtered[~mask]
    
    # Convert traffic_intensity to offered load percentage (0-100%)
    # traffic_intensity ranges from 0.0001 to 0.002 with step 0.0001
    # This corresponds to 0-100% offered load
    df_filtered['offered_load'] = ((df_filtered['traffic_intensity'] - 0.0001) / (0.002 - 0.0001)) * 100
    
    return df_filtered

def create_comparison_plots(df):
    """Create comparison plots for different metrics"""
    
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
    
    # Get unique cluster strategies
    strategies = df['cluster_strategy'].unique()
    
    # Create color map for strategies
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    color_map = dict(zip(strategies, colors))
    
    # Create subplots (changed to 3x3 but with 8 metrics, last subplot will be empty)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Performance Comparison Across Cluster Strategies vs Offered Load', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        
        # Plot each strategy
        for strategy in strategies:
            strategy_data = df[df['cluster_strategy'] == strategy].sort_values('offered_load')
            
            ax.plot(strategy_data['offered_load'], strategy_data[metric], 
                   marker='o', linewidth=2, markersize=6, 
                   label=strategy, color=color_map[strategy])
        
        ax.set_xlabel('Offered Load (%)', fontsize=10)
        ax.set_ylabel(metric_labels[metric], fontsize=10)
        ax.set_title(f'{metric_labels[metric]} vs Offered Load', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
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
    
    strategies = df['cluster_strategy'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    color_map = dict(zip(strategies, colors))
    
    figures = []
    
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        for strategy in strategies:
            strategy_data = df[df['cluster_strategy'] == strategy].sort_values('offered_load')
            
            ax.plot(strategy_data['offered_load'], strategy_data[metric], 
                   marker='o', linewidth=3, markersize=8, 
                   label=strategy, color=color_map[strategy])
        
        ax.set_xlabel('Offered Load (%)', fontsize=12)
        ax.set_ylabel(metric_labels[metric], fontsize=12)
        ax.set_title(f'{metric_labels[metric]} vs Offered Load', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
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
        print(f"Offered load range: {df['offered_load'].min():.1f}% - {df['offered_load'].max():.1f}%")
        
        # Create overview plot with all metrics
        print("\nCreating overview comparison plot...")
        overview_fig = create_comparison_plots(df)
        
        # Create individual plots for better visibility
        print("Creating individual metric plots...")
        individual_figs = create_individual_plots(df)
        
        # Show all plots
        plt.show()
        
        print("\nPlots displayed successfully!")
        print(f"Total figures created: {1 + len(individual_figs)}")
        
    except FileNotFoundError:
        print("Error: Could not find 'equally_2_live_time.xlsx' file.")
        print("Please make sure the file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
