"""
Train and compare multiple timeseries-forecasting models on
arrival_data/ML_tests/sample.csv (minute,function_id,count).

Models:
  1. LSTM           (PyTorch)
  2. Random Forest  (scikit-learn)
  3. XGBoost
  4. LightGBM
  5. Linear Regression (baseline)

The script prints a comparison table (MAE / RMSE / R^2 / MAPE) on the
held-out test split, then saves the best model + everything the simulator
needs to load and predict the next-minute count.

Saved artifacts (in the same folder as this script, under ./best_model/):
  - bundle.joblib   -> dict with model_type, scaler, lookback, feature_names, model (if non-LSTM)
  - lstm.pth        -> only if the winner is LSTM
  - metadata.json   -> human-readable summary

Inference example at the bottom of this file.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")

# ----------------------------- configuration ---------------------------------
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "day_night.csv"
OUT_DIR = HERE / "best_model_day_night"
OUT_DIR.mkdir(exist_ok=True)

LOOKBACK = 120          # use last 360 minutes to predict the next minute
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15        # remainder -> test
RANDOM_STATE = 42
QUANTILE = 0.7         # train an upper-quantile predictor to avoid underestimation

# ----------------------------- data loading ----------------------------------
def load_series(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df.sort_values("minute").reset_index(drop=True)
    if df["function_id"].nunique() > 1:
        # if more than one function, train on the most populous one
        top = df["function_id"].value_counts().idxmax()
        df = df[df["function_id"] == top].reset_index(drop=True)
    series = df["count"].astype(float).to_numpy()
    return series


def chrono_split(n: int, train_frac: float, val_frac: float):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def make_supervised(series: np.ndarray, lookback: int):
    """Return X (N, lookback) and y (N,) using a sliding window."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ----------------------------- metrics ---------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def under_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of predictions that fall below the true value (under-prediction rate)."""
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_pred < y_true) * 100.0)


@dataclass
class Score:
    name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    under: float  # % of under-predictions

    def as_row(self):
        return (
            f"{self.name:<18} {self.mae:>10.4f} {self.rmse:>10.4f} "
            f"{self.r2:>10.4f} {self.mape:>10.4f} {self.under:>10.4f}"
        )


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Score:
    return Score(
        name=name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        mape=mape(y_true, y_pred),
        under=under_rate(y_true, y_pred),
    )


# ----------------------------- model training --------------------------------
def pinball_loss_pytorch(y_true, y_pred, quantile: float):
    """Pinball (quantile) loss for PyTorch."""
    import torch
    diff = y_true - y_pred
    return torch.mean(torch.max(quantile * diff, (quantile - 1.0) * diff))


def get_lstm_model():
    import torch.nn as nn
    class QuantileLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            x, _ = self.lstm2(x)
            x = self.dropout2(x)
            x = x[:, -1, :]
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return QuantileLSTM()


def train_lstm(X_train, y_train, X_val, y_val, lookback, scaler_y):
    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import copy

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train.reshape(-1, lookback, 1), dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val.reshape(-1, lookback, 1), dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = get_lstm_model().to(device)
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float('inf')
    best_weights = None
    patience = 8
    patience_counter = 0

    for epoch in range(60):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = pinball_loss_pytorch(batch_y, preds, QUANTILE)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = pinball_loss_pytorch(y_val_t, val_preds, QUANTILE).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def predict_lstm(model, X, lookback):
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X.reshape(-1, lookback, 1), dtype=torch.float32).to(device)
        preds = model(X_t)
    return preds.cpu().numpy().ravel()


def main():
    series = load_series(CSV_PATH)
    print(f"Loaded {len(series)} samples. Range=[{series.min():.0f}, {series.max():.0f}], mean={series.mean():.2f}")

    n = len(series)
    s_train, s_val, s_test = chrono_split(n, TRAIN_FRAC, VAL_FRAC)

    # Scale on training portion only.
    scaler = MinMaxScaler()
    scaler.fit(series[s_train].reshape(-1, 1))
    scaled = scaler.transform(series.reshape(-1, 1)).ravel()

    X_all, y_all = make_supervised(scaled, LOOKBACK)
    # The first LOOKBACK points have no features, so original index i corresponds to row (i - LOOKBACK).
    train_end = s_train.stop - LOOKBACK
    val_end = s_val.stop - LOOKBACK
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    scores: list[Score] = []
    trained: dict[str, object] = {}
    # Per-model additive bias offset (in *scaled* space) applied after model.predict().
    # Used for models that don't support quantile loss natively (e.g. RandomForest).
    bias_offsets: dict[str, float] = {}

    # ---------------- Quantile Linear Regression ----------------
    # QuantileRegressor minimises pinball loss directly -> biased toward the QUANTILE-th percentile.
    # lr = QuantileRegressor(quantile=QUANTILE, alpha=0.0, solver="highs")
    # lr.fit(X_train, y_train)
    # y_pred = lr.predict(X_test)
    # scores.append(evaluate("QuantileLinear", inv_scale(y_test, scaler), inv_scale(y_pred, scaler)))
    # trained["QuantileLinear"] = lr
    # bias_offsets["QuantileLinear"] = 0.0

    # ---------------- Random Forest ----------------
    # Sklearn's RandomForestRegressor has no quantile loss, so we apply a post-hoc bias
    # correction: shift predictions by the QUANTILE-th percentile of validation residuals.
    # This makes the residual distribution centred above zero, avoiding underestimation.
    print("\nTraining Random Forest with grid search...")
    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, None]}
    best_rf_rmse = float('inf')
    best_rf_model = None
    best_rf_bias = 0.0

    for params in ParameterGrid(rf_param_grid):
        rf = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE, **params)
        rf.fit(X_train, y_train)
        rf_val_residuals = y_val - rf.predict(X_val)
        rf_bias = float(np.quantile(rf_val_residuals, QUANTILE))
        y_pred_val = rf.predict(X_val) + rf_bias
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        if val_rmse < best_rf_rmse:
            best_rf_rmse = val_rmse
            best_rf_model = rf
            best_rf_bias = rf_bias

    rf = best_rf_model
    y_pred = rf.predict(X_test) + best_rf_bias
    scores.append(evaluate("RandomForest", inv_scale(y_test, scaler), inv_scale(y_pred, scaler)))
    trained["RandomForest"] = rf
    bias_offsets["RandomForest"] = best_rf_bias

    # ---------------- XGBoost (quantile regression) ----------------
    try:
        from xgboost import XGBRegressor
        print("Training XGBoost with grid search...")
        xgb_param_grid = {'n_estimators': [200, 500], 'max_depth': [4, 6], 'learning_rate': [0.05, 0.1]}
        best_xgb_rmse = float('inf')
        best_xgb_model = None

        for params in ParameterGrid(xgb_param_grid):
            xgb = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=QUANTILE,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                early_stopping_rounds=20,
                **params
            )
            xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred_val = xgb.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            if val_rmse < best_xgb_rmse:
                best_xgb_rmse = val_rmse
                best_xgb_model = xgb

        xgb = best_xgb_model
        y_pred = xgb.predict(X_test)
        scores.append(evaluate("XGBoost", inv_scale(y_test, scaler), inv_scale(y_pred, scaler)))
        trained["XGBoost"] = xgb
        bias_offsets["XGBoost"] = 0.0
    except ImportError:
        print("xgboost not installed — skipping.")

    # ---------------- LightGBM (quantile regression) ----------------
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        print("Training LightGBM with grid search...")
        lgbm_param_grid = {'n_estimators': [500, 1000], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]}
        best_lgbm_rmse = float('inf')
        best_lgbm_model = None

        for params in ParameterGrid(lgbm_param_grid):
            lgbm = LGBMRegressor(
                objective="quantile",
                alpha=QUANTILE,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params
            )
            lgbm.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(20), log_evaluation(0)],
            )
            y_pred_val = lgbm.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            if val_rmse < best_lgbm_rmse:
                best_lgbm_rmse = val_rmse
                best_lgbm_model = lgbm

        lgbm = best_lgbm_model
        y_pred = lgbm.predict(X_test)
        scores.append(evaluate("LightGBM", inv_scale(y_test, scaler), inv_scale(y_pred, scaler)))
        trained["LightGBM"] = lgbm
        bias_offsets["LightGBM"] = 0.0
    except ImportError:
        print("lightgbm not installed — skipping.")

    # ---------------- LSTM (pinball loss) ----------------
    try:
        lstm = train_lstm(X_train, y_train, X_val, y_val, LOOKBACK, scaler)
        y_pred = predict_lstm(lstm, X_test, LOOKBACK)
        scores.append(evaluate("LSTM", inv_scale(y_test, scaler), inv_scale(y_pred, scaler)))
        trained["LSTM"] = lstm
        bias_offsets["LSTM"] = 0.0
    except ImportError:
        print("torch not installed — skipping LSTM.")

    # ---------------- Report ----------------
    print(
        f"\n=== Test-set comparison "
        f"(quantile={QUANTILE}; Under(%) = fraction of test points where prediction < truth) ==="
    )
    header = (
        f"{'Model':<18} {'MAE':>10} {'RMSE':>10} {'R^2':>10} "
        f"{'MAPE(%)':>10} {'Under(%)':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in sorted(scores, key=lambda s: s.rmse):
        print(s.as_row())

    # ---------------- Pick best by RMSE ----------------
    best_by_rmse = min(scores, key=lambda s: s.rmse)
    print(f"\nBest model by RMSE: {best_by_rmse.name}")

    # ---------------- Choose model to save ----------------
    available_models = [s.name for s in scores]
    print(f"Available models: {', '.join(available_models)}")
    chosen_name = input(f"Enter model to save (default '{best_by_rmse.name}'): ").strip()
    
    if not chosen_name:
        best = best_by_rmse
    else:
        matches = [s for s in scores if s.name.lower() == chosen_name.lower()]
        if matches:
            best = matches[0]
            print(f"Selected {best.name}")
        else:
            print(f"Model '{chosen_name}' not found. Falling back to {best_by_rmse.name}")
            best = best_by_rmse

    # ---------------- Save artifacts ----------------
    bundle = {
        "model_type": best.name,
        "lookback": LOOKBACK,
        "scaler": scaler,
        "quantile": QUANTILE,
        "bias_offset": bias_offsets.get(best.name, 0.0),
        "score": asdict(best),
        # Inference contract:
        #   1. take the last `lookback` raw counts as a 1-D array of length lookback.
        #   2. scale with bundle["scaler"] (MinMaxScaler).
        #   3. for non-LSTM: feed shape (1, lookback) into bundle["model"].predict(...).
        #      for LSTM: load lstm.pth from same folder, feed shape (1, lookback, 1).
        #   4. add bundle["bias_offset"] (scaled-space additive correction; 0 for quantile-trained models).
        #   5. inverse-transform the corrected scalar prediction with bundle["scaler"].
    }

    if best.name == "LSTM":
        import torch
        torch_path = OUT_DIR / "lstm.pth"
        torch.save(trained["LSTM"].state_dict(), torch_path)
        bundle["keras_path"] = "lstm.pth"
    else:
        bundle["model"] = trained[best.name]

    bundle_path = OUT_DIR / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"Saved bundle -> {bundle_path}")

    metadata = {
        "best_model": best.name,
        "lookback": LOOKBACK,
        "csv": CSV_PATH.name,
        "n_samples": int(n),
        "metrics": {s.name: asdict(s) for s in scores},
        "files": (
            ["bundle.joblib", "lstm.pth"] if best.name == "LSTM" else ["bundle.joblib"]
        ),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata -> {OUT_DIR / 'metadata.json'}")


def inv_scale(arr: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(arr).reshape(-1, 1)).ravel()


# ----------------------------- inference helper ------------------------------
def load_predictor(bundle_dir: str | Path):
    """
    Returns a callable predict(last_window) -> float.
    `last_window` must be a 1-D iterable of length `bundle['lookback']`
    containing the most recent raw `count` values.
    """
    bundle_dir = Path(bundle_dir)
    bundle = joblib.load(bundle_dir / "bundle.joblib")
    scaler = bundle["scaler"]
    lookback = bundle["lookback"]
    bias_offset = float(bundle.get("bias_offset", 0.0))

    if bundle["model_type"] == "LSTM":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keras_model = get_lstm_model().to(device)
        keras_model.load_state_dict(torch.load(bundle_dir / bundle["keras_path"], map_location=device))
        keras_model.eval()

        def predict(window):
            arr = np.asarray(window, dtype=float).reshape(-1, 1)
            assert arr.shape[0] == lookback, f"need {lookback} points, got {arr.shape[0]}"
            scaled = scaler.transform(arr).ravel().reshape(1, lookback, 1)
            with torch.no_grad():
                scaled_t = torch.tensor(scaled, dtype=torch.float32).to(device)
                scaled_pred = keras_model(scaled_t).item() + bias_offset
            return float(scaler.inverse_transform([[scaled_pred]])[0, 0])
    else:
        model = bundle["model"]
        # LightGBM stores anonymous feature names when fit on numpy; calling
        # the booster directly bypasses sklearn's feature-name check warning.
        booster = getattr(model, "booster_", None)

        def predict(window):
            arr = np.asarray(window, dtype=float).reshape(-1, 1)
            assert arr.shape[0] == lookback, f"need {lookback} points, got {arr.shape[0]}"
            scaled = scaler.transform(arr).ravel().reshape(1, lookback)
            raw = booster.predict(scaled) if booster is not None else model.predict(scaled)
            scaled_pred = float(np.ravel(raw)[0]) + bias_offset
            return float(scaler.inverse_transform([[scaled_pred]])[0, 0])

    return predict, bundle


if __name__ == "__main__":
    main()
    # Smoke-test the saved bundle:
    predict, meta = load_predictor(OUT_DIR)
    last = load_series(CSV_PATH)[-meta["lookback"]:]
    print(f"\nSmoke test — next-minute prediction using last {meta['lookback']} samples: {predict(last):.2f}")
