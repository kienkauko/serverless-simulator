import matplotlib.pyplot as plt
import numpy as np

# Font and size definitions
fontsize = 14
legend_size = 14
tick_size = 14

# Data
requests_per_second = [491, 694, 867, 1195, 1428, 1603]
latency_centralized = [0.186, 0.244, 0.336, 0.592, 2.2, 3.604]
latency_edge_cloud = [0.16, 0.186, 0.248, 0.366, 1.24, 1.44]

# Plotting option: 'both' or 'c'
plot_option = 'both'

x_pos = np.arange(len(requests_per_second))
bar_width = 0.35

# Create the bar chart
plt.figure(figsize=(10, 6))

if plot_option == 'centralized_only':
    plt.bar(x_pos, latency_centralized, width=bar_width, label='Centralized Cloud', align='center', alpha=0.7)
elif plot_option == 'both':
    plt.bar(x_pos - bar_width/2, latency_centralized, width=bar_width, label='Centralized Cloud', align='center', alpha=0.7)
    plt.bar(x_pos + bar_width/2, latency_edge_cloud, width=bar_width, label='Edge-Cloud', align='center', alpha=0.7)


# Add labels and title
plt.xlabel('Request per Second', fontsize=fontsize)
plt.ylabel('Latency caused by Network (s)', fontsize=fontsize)
# plt.title('Latency vs. Request per Second', fontsize=fontsize)
plt.xticks(x_pos, requests_per_second, fontsize=tick_size) # Set x-axis labels to the actual request values
plt.yticks(fontsize=tick_size)
plt.legend(fontsize=legend_size)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()