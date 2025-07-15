import numpy as np
import matplotlib.pyplot as plt

# Pseudo-data generation
apps = ["light", "medium", "heavy"]
configs = ["always_on", "always_off", "adaptive"]
lambdas = [1.0, 5.0, 10.0, 15.0, 20.0]

# Generate pseudo-data (replace with your actual data)
np.random.seed(42)
data = {
    "blocking_prob": {
        app: {
            "always_on": np.round(np.linspace(0.1, 0.8, len(lambdas)) * (i + 1), 3),
            "always_off": np.round(np.linspace(0.3, 0.9, len(lambdas)) * (i + 1), 3),
            "adaptive": np.round(np.linspace(0.05, 0.5, len(lambdas)) * (i + 1), 3)
        } for i, app in enumerate(apps)
    },
    "latency": {
        app: {
            "always_on": np.round(np.linspace(5, 50, len(lambdas)) * (i + 1), 1),
            "always_off": np.round(np.linspace(10, 60, len(lambdas)) * (i + 1), 1),
            "adaptive": np.round(np.linspace(2, 30, len(lambdas)) * (i + 1), 1)
        } for i, app in enumerate(apps)
    },
    "resource_consumption": {
        app: {
            "always_on": np.round(np.linspace(20, 90, len(lambdas)) * (i + 1), 1),
            "always_off": np.round(np.linspace(10, 50, len(lambdas)) * (i + 1), 1),
            "adaptive": np.round(np.linspace(15, 70, len(lambdas)) * (i + 1), 1)
        } for i, app in enumerate(apps)
    }
}

# Plotting
metrics = {
    "blocking_prob": "Blocking Probability",
    "latency": "Latency (ms)",
    "resource_consumption": "Resource Consumption (%)"
}
colors = {
    "always_on": "red",
    "always_off": "blue",
    "adaptive": "green"
}

for metric, ylabel in metrics.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{ylabel} by Application", fontsize=14, y=1.05)
    
    for i, app in enumerate(apps):
        for config in configs:
            axes[i].plot(
                lambdas, 
                data[metric][app][config], 
                label=config, 
                color=colors[config],
                marker="o",
                linestyle="--" if config == "always_off" else "-"
            )
        axes[i].set_title(f"Application: {app}")
        axes[i].set_xlabel("Arrival Rate (λ)")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()