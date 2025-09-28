import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Plotting Configuration ---
fontsize = 14
ticksize = 12
legendsize = 12
# output_folder = 'charts'

# --- Data Loading and Processing ---
# TODO: Replace with the actual path to your data file.
# The file should be in CSV format.
script_dir = os.path.dirname(__file__)
data_file_path = os.path.join(script_dir, 'simulation_results.xlsx')

# Define metrics to plot. These should match column names in your file.
metrics_to_plot = [
    'blocking_percentage',
    'avg_offloaded_to_cloud',
    'avg_total_latency',
    'avg_spawn_time',
    'avg_processing_time',
    'avg_network_time'
]

# Define pretty names for y-axis labels
y_axis_labels = {
    'blocking_percentage': 'Blocking Percentage (%)',
    'avg_offloaded_to_cloud': 'Offloaded to Cloud (%)',
    'avg_total_latency': 'Average Total Latency (s)',
    'avg_spawn_time': 'Average Spawn Time (s)',
    'avg_processing_time': 'Average Processing Time (s)',
    'avg_network_time': 'Average Network Time (s)'
}

# Define plot styles for each scenario
plot_styles = {
    'centralized_cloud': {'color': 'r', 'marker': 'o', 'linestyle': '-'},
    'massive_edge_cloud': {'color': 'g', 'marker': 's', 'linestyle': '--'},
    'massive_edge': {'color': 'b', 'marker': '^', 'linestyle': ':'}
}


# --- Plotting ---
try:
    # Read Excel file instead of CSV
    df = pd.read_excel(data_file_path) 
    # Let's rename the first column to 'scenario' for easier access
    df.rename(columns={df.columns[0]: 'scenario'}, inplace=True)

    scenarios = df['scenario'].unique()

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    #     print(f"Created directory: {output_folder}")

    for metric in metrics_to_plot:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in the data file. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        
        for scenario in scenarios:
            scenario_df = df[df['scenario'] == scenario]
            style = plot_styles.get(scenario, {'marker': 'o', 'linestyle': '-'}) # Default style
            plt.plot(scenario_df['traffic_intensity'], scenario_df[metric], label=scenario, **style)

        plt.xlabel('Load (Traffic Intensity)', fontsize=fontsize)
        plt.ylabel(y_axis_labels.get(metric, metric), fontsize=fontsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.legend(fontsize=legendsize)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        # Save the figure
        # output_path = os.path.join(output_folder, f"{metric}.png")
        # plt.savefig(output_path)
        # print(f"Saved plot to {output_path}")
        
        # plt.close() # Close the figure to free up memory

except FileNotFoundError:
    print(f"Error: Data file not found at '{data_file_path}'.")
    print("Please update the 'data_file_path' variable with the correct location of your Excel file.")
except Exception as e:
    print(f"An error occurred: {e}")
