import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

def load_and_process_data(edge_server_number):
    """Load data from both Excel files and process according to requirements"""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load both Excel files
    equally_file = os.path.join(script_dir, 'equally_2_live_time.xlsx')
    population_file = os.path.join(script_dir, 'population_2_live_time.xlsx')
    
    df_equally = pd.read_excel(equally_file)
    df_population = pd.read_excel(population_file)
    
    # Add mapping strategy labels
    df_equally['mapping_strategy'] = 'Equally'
    df_population['mapping_strategy'] = 'Population'
    
    # Combine both datasets
    df_combined = pd.concat([df_equally, df_population], ignore_index=True)
    
    # Filter data: exclude centralized_cloud, only keep massive_edge_cloud and massive_edge
    df_filtered = df_combined[df_combined['cluster_strategy'].isin(['massive_edge_cloud', 'massive_edge'])].copy()
    
    # Filter by selected edge server number
    df_filtered = df_filtered[df_filtered['edge_server_number'] == edge_server_number].copy()
    
    # Convert traffic_intensity to offered load percentage (0-100%)
    # traffic_intensity ranges from 0.0001 to 0.002 with step 0.0001
    # This corresponds to 0-100% offered load
    df_filtered['offered_load'] = ((df_filtered['traffic_intensity'] - 0.0001) / (0.002 - 0.0001)) * 100
    
    # Create a combined label for strategy and mapping
    df_filtered['strategy_mapping'] = df_filtered['cluster_strategy'] + '_' + df_filtered['mapping_strategy']
    
    return df_filtered

def create_comparison_plots(df, edge_server_number):
    """Create comparison plots for different metrics comparing mapping strategies"""
    
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
    
    # Get unique strategy-mapping combinations
    strategy_mappings = df['strategy_mapping'].unique()
    
    # Create color and style maps
    colors = ['blue', 'red', 'green', 'orange']  # Fixed colors for consistency
    color_map = {}
    linestyles = {}
    markers = {}
    
    color_idx = 0
    for combo in sorted(strategy_mappings):
        color_map[combo] = colors[color_idx % len(colors)]
        
        if 'massive_edge_cloud' in combo:
            if 'Equally' in combo:
                linestyles[combo] = '-'
                markers[combo] = 'o'
            else:  # Population
                linestyles[combo] = '-'
                markers[combo] = '^'
        else:  # massive_edge
            if 'Equally' in combo:
                linestyles[combo] = '--'
                markers[combo] = 's'
            else:  # Population
                linestyles[combo] = '--'
                markers[combo] = 'D'
        
        color_idx += 1
    
    # Create subplots (3x3 grid with 8 metrics)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Mapping Strategy Comparison with {edge_server_number} Edge Servers\n(Equally vs Population)', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        
        # Plot each strategy-mapping combination
        for combo in sorted(strategy_mappings):
            combo_data = df[df['strategy_mapping'] == combo].sort_values('offered_load')
            
            # Extract strategy and mapping for legend
            if 'massive_edge_cloud' in combo:
                strategy_name = "Massive Edge Cloud"
            else:  # massive_edge
                strategy_name = "Massive Edge"
            
            if 'Equally' in combo:
                mapping_name = "Equally"
            else:  # Population
                mapping_name = "Population"
            
            label = f"{strategy_name} ({mapping_name})"
            
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

def create_individual_plots(df, edge_server_number):
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
    
    strategy_mappings = df['strategy_mapping'].unique()
    
    # Create color and style maps
    colors = ['blue', 'red', 'green', 'orange']  # Fixed colors for consistency
    color_map = {}
    linestyles = {}
    markers = {}
    
    color_idx = 0
    for combo in sorted(strategy_mappings):
        color_map[combo] = colors[color_idx % len(colors)]
        
        if 'massive_edge_cloud' in combo:
            if 'Equally' in combo:
                linestyles[combo] = '-'
                markers[combo] = 'o'
            else:  # Population
                linestyles[combo] = '-'
                markers[combo] = '^'
        else:  # massive_edge
            if 'Equally' in combo:
                linestyles[combo] = '--'
                markers[combo] = 's'
            else:  # Population
                linestyles[combo] = '--'
                markers[combo] = 'D'
        
        color_idx += 1
    
    figures = []
    
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        for combo in sorted(strategy_mappings):
            combo_data = df[df['strategy_mapping'] == combo].sort_values('offered_load')
            
            # Extract strategy and mapping for legend
            if 'massive_edge_cloud' in combo:
                strategy_name = "Massive Edge Cloud"
            else:  # massive_edge
                strategy_name = "Massive Edge"
            
            if 'Equally' in combo:
                mapping_name = "Equally"
            else:  # Population
                mapping_name = "Population"
            
            label = f"{strategy_name} ({mapping_name})"
            
            ax.plot(combo_data['offered_load'], combo_data[metric], 
                   marker=markers[combo], linewidth=3, markersize=8, 
                   linestyle=linestyles[combo], label=label, color=color_map[combo])
        
        ax.set_xlabel('Offered Load (%)', fontsize=14)
        ax.set_ylabel(metric_labels[metric], fontsize=14)
        ax.set_title(f'{metric_labels[metric]} vs Offered Load\n({edge_server_number} Edge Servers)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only show legend for blocking percentage plot
        if metric == 'blocking_percentage':
            ax.legend(fontsize=12, loc='best')
        
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def create_strategy_specific_plots(df, edge_server_number):
    """Create plots comparing mapping strategies for each cluster strategy separately"""
    
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
        fig.suptitle(f'Mapping Strategy Comparison - {strategy_title}\n({edge_server_number} Edge Servers)', 
                     fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes_flat[i]
            
            # Plot each mapping strategy for this cluster strategy
            for mapping in ['Equally', 'Population']:
                mapping_data = df_strategy[df_strategy['mapping_strategy'] == mapping].sort_values('offered_load')
                
                if len(mapping_data) > 0:
                    color = 'blue' if mapping == 'Equally' else 'red'
                    marker = 'o' if mapping == 'Equally' else '^'
                    linestyle = '-'
                    
                    ax.plot(mapping_data['offered_load'], mapping_data[metric], 
                           marker=marker, linewidth=2, markersize=6, 
                           linestyle=linestyle, label=f"{mapping} Mapping", 
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

def get_available_edge_numbers():
    """Get available edge server numbers from the Excel files"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        equally_file = os.path.join(script_dir, 'equally_2_live_time.xlsx')
        
        df = pd.read_excel(equally_file)
        df_filtered = df[df['cluster_strategy'].isin(['massive_edge_cloud', 'massive_edge'])]
        available_numbers = sorted(df_filtered['edge_server_number'].unique())
        return available_numbers
    except Exception as e:
        print(f"Error reading file: {e}")
        return [3000, 5000, 7000]  # Default values

def main():
    """Main function to run the analysis"""
    try:
        # Get available edge server numbers
        available_numbers = get_available_edge_numbers()
        
        print("Mapping Strategy Comparison Analysis")
        print("=" * 50)
        print("Available edge server numbers:", available_numbers)
        print()
        
        # Get user input for edge server number
        while True:
            try:
                user_input = input(f"Enter edge server number to analyze {available_numbers}: ")
                edge_server_number = int(user_input)
                
                if edge_server_number in available_numbers:
                    break
                else:
                    print(f"Please choose from available numbers: {available_numbers}")
            except ValueError:
                print("Please enter a valid number.")
        
        print(f"\nAnalyzing data with {edge_server_number} edge servers...")
        
        # Load and process data
        df = load_and_process_data(edge_server_number)
        
        if len(df) == 0:
            print(f"No data found for {edge_server_number} edge servers.")
            return
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Cluster strategies found: {df['cluster_strategy'].unique()}")
        print(f"Mapping strategies found: {df['mapping_strategy'].unique()}")
        print(f"Strategy-mapping combinations: {sorted(df['strategy_mapping'].unique())}")
        print(f"Offered load range: {df['offered_load'].min():.1f}% - {df['offered_load'].max():.1f}%")
        
        # Create overview plot comparing all strategy-mapping combinations
        print("\nCreating overview comparison plot...")
        overview_fig = create_comparison_plots(df, edge_server_number)
        
        # Create individual plots for better visibility
        print("Creating individual metric plots...")
        individual_figs = create_individual_plots(df, edge_server_number)
        
        # Create strategy-specific comparison plots
        print("Creating strategy-specific comparison plots...")
        strategy_figs = create_strategy_specific_plots(df, edge_server_number)
        
        # Show all plots
        plt.show()
        
        total_figs = 1 + len(individual_figs) + len(strategy_figs)
        print(f"\nPlots displayed successfully!")
        print(f"Total figures created: {total_figs}")
        print(f"- 1 overview plot")
        print(f"- {len(individual_figs)} individual metric plots")
        print(f"- {len(strategy_figs)} strategy-specific comparison plots")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find Excel file. {str(e)}")
        print("Please make sure both 'equally_2_live_time.xlsx' and 'population_2_live_timexx.xlsx' files are in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
