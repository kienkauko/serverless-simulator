import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# --- error metric functions ---
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mpe(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- load & pivot data ---
# Get the current directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create path to the csv file in the current directory
csv_path = os.path.join(current_dir, "markov_vs_sim_D.csv")
df = pd.read_csv(csv_path)

# pivot so that for each case_id we have Markov vs Simulator columns
metrics = ["blocking_probability", "latency", "cpu_usage", "ram_usage"]
# round metric values to two decimal places before comparison
df[metrics] = df[metrics].round(2)
pivot = df.pivot(index="case_id", columns="model_type", values=metrics)

# drop any incomplete cases
pivot = pivot.dropna()

# --- compute and display errors ---
results = []
for metric in metrics:
    y_true = pivot[(metric, "Simulator")].values
    y_pred = pivot[(metric, "Markov")].values

    results.append({
        "metric": metric,
        "MAPE(%)": mape(y_true, y_pred),
        "MPE(%)": mpe(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    })

error_df = pd.DataFrame(results).set_index("metric")
print(error_df.round(4))