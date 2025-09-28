import pandas as pd
import matplotlib.pyplot as plt
import os

# (Assuming similar setup code as congested_links.py for data loading and colors)
script_dir = os.path.dirname(__file__)
data_file_path = os.path.join(script_dir, 'simulation_results.xlsx')
congestion_columns = ['3-3', '3-2', '2-2', '2-1', '1-1', '1-0', '0-0']
congestion_colors = {
    '3-3': '#d62728', '3-2': '#ff7f0e', '2-2': '#ffbb78', '2-1': '#2ca02c',
    '1-1': '#98df8a', '1-0': '#1f77b4', '0-0': '#aec7e8'
}

def plot_congestion_area_chart(scenarios=None):
    df = pd.read_excel(data_file_path, sheet_name='Congestion_Results')
    df.rename(columns={df.columns[0]: 'scenario'}, inplace=True)

    if scenarios:
        df = df[df['scenario'].isin(scenarios)]

    unique_scenarios = df['scenario'].unique()
    
    # Create a subplot for each scenario
    fig, axes = plt.subplots(nrows=len(unique_scenarios), ncols=1, 
                             figsize=(10, 6 * len(unique_scenarios)), sharex=True)
    if len(unique_scenarios) == 1:
        axes = [axes] # Make it iterable

    for ax, scenario in zip(axes, unique_scenarios):
        scenario_df = df[df['scenario'] == scenario].sort_values('traffic_intensity')
        
        # Calculate percentages
        totals = scenario_df[congestion_columns].sum(axis=1)
        percentages = scenario_df[congestion_columns].div(totals, axis=0) * 100
        
        # Plot stacked area
        ax.stackplot(scenario_df['traffic_intensity'], percentages.T, 
                     labels=congestion_columns, colors=[congestion_colors[c] for c in congestion_columns])
        
        ax.set_title(f'Congestion for: {scenario}', fontsize=14)
        ax.set_ylabel('Congestion Percentage (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('Traffic Intensity (Load)', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Congestion Level', loc='upper right')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.show()

# Example usage:
plot_congestion_area_chart(scenarios=['centralized_cloud', 'massive_edge_cloud'])