import pandas as pd
import matplotlib.pyplot as plt
import os

# (Assuming similar setup code as congested_links.py for data loading and colors)
script_dir = os.path.dirname(__file__)
data_file_path = os.path.join(script_dir, 'equally_2_live_time.xlsx')
congestion_columns = ['3-3', '3-2', '2-2', '2-1', '1-1', '1-0', '0-0']
congestion_colors = {
    '3-3': '#d62728', '3-2': '#ff7f0e', '2-2': '#ffbb78', '2-1': '#2ca02c',
    '1-1': '#98df8a', '1-0': '#1f77b4', '0-0': '#aec7e8'
}

def plot_congestion_area_chart(edge_server_number):
    df = pd.read_excel(data_file_path, sheet_name='Congestion_Results')
    df.rename(columns={df.columns[0]: 'cluster_strategy'}, inplace=True)

    # Filter scenarios: include centralized_cloud (regardless of edge_server_number) 
    # and massive_edge scenarios with specified edge server number
    df = df[((df['cluster_strategy'] == 'centralized_cloud') | 
            ((df['cluster_strategy'].isin(['massive_edge_cloud', 'massive_edge'])) & 
             (df['edge_server_number'] == edge_server_number)))]

    unique_scenarios = df['cluster_strategy'].unique()
    
    # Create individual plots for each scenario
    for scenario in unique_scenarios:
        scenario_df = df[df['cluster_strategy'] == scenario].sort_values('traffic_intensity')
        
        # Calculate percentages
        totals = scenario_df[congestion_columns].sum(axis=1)
        percentages = scenario_df[congestion_columns].div(totals, axis=0) * 100
        
        # Convert traffic intensity to percentage (0.0001 to 0.002 -> 0% to 100%)
        min_intensity = scenario_df['traffic_intensity'].min()
        max_intensity = scenario_df['traffic_intensity'].max()
        offered_load_percent = ((scenario_df['traffic_intensity'] - min_intensity) / 
                               (max_intensity - min_intensity)) * 100
        
        # Create individual figure
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        
        # Plot stacked area
        ax.stackplot(offered_load_percent, percentages.T, 
                     labels=congestion_columns, colors=[congestion_colors[c] for c in congestion_columns])
        
        ax.set_title(f'Scenario: {scenario}', fontsize=14)
        ax.set_xlabel('Offered Load (%)', fontsize=12)
        ax.set_ylabel('Percentage of Bottleneck by Link (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title='Link layer-layer', loc='upper right')
        
        plt.tight_layout()
        plt.show()

# Example usage:
plot_congestion_area_chart(edge_server_number=5000)