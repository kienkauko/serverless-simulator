import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Read data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "2026_04_17_1021_standalone_processing_time.csv")
df = pd.read_csv(csv_path)
data = df["processing_time_s"].dropna().values

print(f"Loaded {len(data)} processing time samples")
print(f"Min: {data.min():.4f}s, Max: {data.max():.4f}s, Mean: {data.mean():.4f}s, Std: {data.std():.4f}s")

# Candidate distributions to fit
candidates = {
    "norm": stats.norm,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "expon": stats.expon,
    "weibull_min": stats.weibull_min,
    "beta": stats.beta,
    "invgauss": stats.invgauss,
    "burr": stats.burr,
    # Heavy-tail distributions
    "pareto": stats.pareto,
    "genpareto": stats.genpareto,
    "levy": stats.levy,
    "cauchy": stats.cauchy,
    "t": stats.t,
    "alpha": stats.alpha,
    "genextreme": stats.genextreme,
    "loggamma": stats.loggamma,
    "fisk": stats.fisk,          # log-logistic
    "burr12": stats.burr12,
}

# Fit each distribution and compute KS test
results = []
fitted_params = {}
for name, dist in candidates.items():
    try:
        params = dist.fit(data)
        ks_stat, p_value = stats.kstest(data, name, args=params)
        results.append((name, ks_stat, p_value, params))
        fitted_params[name] = (dist, params)
        print(f"{name:15s}  KS={ks_stat:.4f}  p={p_value:.4f}  params={tuple(round(p, 4) for p in params)}")
    except Exception as e:
        print(f"{name:15s}  FAILED: {e}")

# Sort by KS statistic (lower is better)
results.sort(key=lambda x: x[1])
best_name, best_ks, best_p, best_params = results[0]
best_dist = candidates[best_name]

print(f"\nBest fit: {best_name} (KS={best_ks:.4f}, p={best_p:.4f})")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(data.min() * 0.9, data.max() * 1.1, 500)

# Left: histogram + top 3 fitted PDFs
ax = axes[0]
ax.hist(data, bins=60, density=True, alpha=0.5, color="steelblue", edgecolor="white", label="Data histogram")
for name, ks, p, params in results[:3]:
    dist = candidates[name]
    pdf = dist.pdf(x, *params)
    ax.plot(x, pdf, linewidth=2, label=f"{name} (KS={ks:.4f})")
ax.set_xlabel("Processing time (s)")
ax.set_ylabel("Density")
ax.set_title("Processing Time Distribution Fitting")
ax.legend()

# Right: scatter of all processing times in order
ax = axes[1]
ax.scatter(range(len(data)), data, s=1, alpha=0.4, color="steelblue")
ax.axhline(data.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean = {data.mean():.3f}s")
ax.set_xlabel("Request index")
ax.set_ylabel("Processing time (s)")
ax.set_title("All Processing Times (5000 requests)")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "processing_time_fit.png"), dpi=150)
plt.show()

print(f"\nFigure saved to processing_time_fit.png")
