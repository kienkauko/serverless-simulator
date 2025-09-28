import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Plotting Configuration ---
fontsize = 14
ticksize = 12
legendsize = 12

# --- Data Loading and Processing ---
script_dir = os.path.dirname(__file__)
data_file_path = os.path.join(script_dir, 'simulation_results.xlsx')

# Define congestion level columns
congestion_columns = ['3-3', '3-2', '2-2', '2-1', '1-1', '1-0', '0-0']

# Define colors for each congestion level
congestion_colors = {
    '3-3': '#d62728',  # Red
    '3-2': '#ff7f0e',  # Orange
    '2-2': '#ffbb78',  # Light orange
    '2-1': '#2ca02c',  # Green
    '1-1': '#98df8a',  # Light green
    '1-0': '#1f77b4',  # Blue
    '0-0': '#aec7e8'   # Light blue
}

def plot_congestion_chart(scenarios=None, show_chart=True):
    """
    Plot stacked bar chart showing congestion percentages.
    
    Parameters:
    scenarios (list): List of scenarios to include. If None, all scenarios are included.
    show_chart (bool): Whether to display the chart or just save it.
    """
    
    try:
        # Read Excel file from Congestion_Results sheet
        df = pd.read_excel(data_file_path, sheet_name='Congestion_Results')
        
        # Rename first two columns for easier access
        df.rename(columns={df.columns[0]: 'scenario'}, inplace=True)
        
        # Filter scenarios if specified
        if scenarios is not None:
            df = df[df['scenario'].isin(scenarios)]
        
        # Get unique scenarios and traffic intensities
        unique_scenarios = df['scenario'].unique()
        unique_traffic_intensities = sorted(df['traffic_intensity'].unique())
        
        # Calculate total congestion for each row
        df['total_congestion'] = df[congestion_columns].sum(axis=1)
        
        # Calculate percentages for each congestion level
        for col in congestion_columns:
            df[f'{col}_percentage'] = (df[col] / df['total_congestion']) * 100
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up bar positions
        n_scenarios = len(unique_scenarios)
        bar_width = 0.8 / n_scenarios if n_scenarios > 1 else 0.8
        x_positions = np.arange(len(unique_traffic_intensities))
        
        # Plot stacked bars for each scenario
        for i, scenario in enumerate(unique_scenarios):
            scenario_data = df[df['scenario'] == scenario].sort_values('traffic_intensity')
            
            # Calculate bar positions for this scenario
            if n_scenarios > 1:
                bar_x = x_positions + (i - (n_scenarios-1)/2) * bar_width
            else:
                bar_x = x_positions
            
            # Initialize bottom values for stacking
            bottom_values = np.zeros(len(scenario_data))
            
            # Plot each congestion level
            for j, col in enumerate(congestion_columns):
                percentage_col = f'{col}_percentage'
                values = scenario_data[percentage_col].values
                
                # Only show legend for first scenario to avoid duplicates
                label = col if i == 0 else ""
                
                ax.bar(bar_x, values, bar_width, bottom=bottom_values, 
                      color=congestion_colors[col], label=label, alpha=0.8)
                
                bottom_values += values
            
            # Add scenario labels if multiple scenarios
            if n_scenarios > 1:
                # Add scenario name below bars
                for j, x in enumerate(bar_x):
                    if j == len(bar_x) // 2:  # Only label middle bar for each scenario
                        ax.text(x, -5, scenario, ha='center', va='top', 
                               fontsize=ticksize-2, rotation=0)
        
        # Customize the plot
        ax.set_xlabel('Traffic Intensity (Load)', fontsize=fontsize)
        ax.set_ylabel('Congestion Percentage (%)', fontsize=fontsize)
        ax.set_title('Network Congestion Distribution by Traffic Intensity', fontsize=fontsize+2)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(unique_traffic_intensities, fontsize=ticksize)
        ax.tick_params(axis='y', labelsize=ticksize)
        
        # Set y-axis to show 0-100%
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        
        # Add legend
        ax.legend(title='Congestion Level', fontsize=legendsize, 
                 title_fontsize=legendsize, loc='upper right')
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if show_chart:
            plt.show()
        
        return fig, ax
        
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_file_path}'.")
        print("Please update the 'data_file_path' variable with the correct location of your Excel file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Example usage:
if __name__ == "__main__":
    # Plot all scenarios
    # print("Plotting all scenarios...")
    # plot_congestion_chart()
    
    # Plot specific scenarios
    print("\nPlotting specific scenarios...")
    plot_congestion_chart(scenarios=['centralized_cloud', 'massive_edge_cloud'])
    
    # Plot single scenario
    # print("\nPlotting single scenario...")
    # plot_congestion_chart(scenarios=['centralized_cloud'])