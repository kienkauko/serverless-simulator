"""
Train and compare multiple timeseries-forecasting models on
arrival_data/ML_tests/sample_2.csv (system logs over ~30 days).

Goal: predict p99_active (p99 of active containers) for the next minute,
given the last `LOOKBACK` minutes of (minute, arrival, p99_active).

Models:
  1. LSTM           (PyTorch)
  2. Random Forest  (scikit-learn)
  3. XGBoost
  4. LightGBM

All models are trained with a quantile / pinball objective (or post-hoc
quantile bias correction for RF) so they bias toward an upper quantile and
avoid systematically underestimating the number of active containers.

Saved artifacts (in the same folder as this script, under ./best_model_active/):
  - bundle.joblib   -> dict with model_type, scalers, lookback, feature_names, model (if non-LSTM)
  - lstm.pth        -> only if the winner is LSTM
  - metadata.json   -> human-readable summary
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")

# ----------------------------- configuration ---------------------------------
HERE = Path(__file__).resolve().parent

TRAFFIC_TYPE = "day_night"   # "day_night" | "non_station"

if TRAFFIC_TYPE == "day_night":
    CSV_PATH = HERE / "day_night_for_train.csv"
    OUT_DIR = HERE / "best_model_active_day_night"
elif TRAFFIC_TYPE == "non_station":
    CSV_PATH = HERE / "non_station_for_train.csv"
    OUT_DIR = HERE / "best_model_active_non_station"
else:
    raise ValueError(f"Unknown TRAFFIC_TYPE: {TRAFFIC_TYPE!r}")
OUT_DIR.mkdir(exist_ok=True)

FEATURE_COLS = ["minute", "arrival", "p99_active"]
TARGET_COL = "p99_active"

LOOKBACK = 120          # use last LOOKBACK minutes to predict the next minute
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15         # remainder -> test
RANDOM_STATE = 42
QUANTILE = 0.5          # upper-quantile predictor to avoid underestimation


# ----------------------------- data loading ----------------------------------
def load_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("minute").reset_index(drop=True)
    # If there are multiple repetitions/seeds, keep the most populous one.
    if "repetition" in df.columns and df["repetition"].nunique() > 1:
        top_rep = df["repetition"].value_counts().idxmax()
        df = df[df["repetition"] == top_rep].reset_index(drop=True)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURE_COLS].astype(float).reset_index(drop=True)


def chrono_split(n: int, train_frac: float, val_frac: float):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def make_supervised(data: np.ndarray, lookback: int, target_idx: int):
    """
    Build sliding windows.
    data: (N, n_features). Returns X (M, lookback, n_features), y (M,).
    The target is data[i, target_idx] for window ending at i-1.
    """
    n = len(data)
    X, y = [], []
    for i in range(lookback, n):
        X.append(data[i - lookback:i, :])
        y.append(data[i, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ----------------------------- metrics ---------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def under_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of predictions that fall below the true value."""
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
    under: float

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


# ----------------------------- RNN (LSTM / GRU) ------------------------------
def pinball_loss_pytorch(y_true, y_pred, quantile: float):
    import torch
    diff = y_true - y_pred
    return torch.mean(torch.max(quantile * diff, (quantile - 1.0) * diff))


def get_lstm_model(n_features: int):
    import torch.nn as nn

    class QuantileLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)
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


def get_gru_model(n_features: int):
    # GRU: same capacity as LSTM but fewer parameters (no cell state) — often trains faster
    import torch.nn as nn

    class QuantileGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru1 = nn.GRU(input_size=n_features, hidden_size=64, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.gru2 = nn.GRU(input_size=64, hidden_size=32, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            x, _ = self.gru1(x)
            x = self.dropout1(x)
            x, _ = self.gru2(x)
            x = self.dropout2(x)
            x = x[:, -1, :]
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    return QuantileGRU()


def _train_rnn(model_fn, X_train, y_train, X_val, y_val, n_features):
    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import copy

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

    model = model_fn(n_features).to(device)
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for _ in range(60):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = pinball_loss_pytorch(batch_y, model(batch_X), QUANTILE)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = pinball_loss_pytorch(y_val_t, model(X_val_t), QUANTILE).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def train_lstm(X_train, y_train, X_val, y_val, n_features):
    return _train_rnn(get_lstm_model, X_train, y_train, X_val, y_val, n_features)


def train_gru(X_train, y_train, X_val, y_val, n_features):
    return _train_rnn(get_gru_model, X_train, y_train, X_val, y_val, n_features)


def predict_rnn(model, X):
    import torch
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(device))
    return preds.cpu().numpy().ravel()


# ----------------------------- main ------------------------------------------
def main():
    df = load_frame(CSV_PATH)
    print(
        f"Loaded {len(df)} samples. "
        f"p99_active range=[{df[TARGET_COL].min():.0f}, {df[TARGET_COL].max():.0f}], "
        f"mean={df[TARGET_COL].mean():.2f}"
    )

    data = df.to_numpy(dtype=float)  # shape (N, n_features)
    n, n_features = data.shape
    target_idx = FEATURE_COLS.index(TARGET_COL)

    s_train, s_val, s_test = chrono_split(n, TRAIN_FRAC, VAL_FRAC)

    # Per-feature MinMax scaling fit on training portion only.
    scaler_X = MinMaxScaler()
    scaler_X.fit(data[s_train])
    scaled = scaler_X.transform(data)

    # A separate target-only scaler so we can inverse-transform predictions
    # without having to pad with the other feature columns.
    scaler_y = MinMaxScaler()
    scaler_y.fit(data[s_train, target_idx].reshape(-1, 1))

    X_all, y_all = make_supervised(scaled, LOOKBACK, target_idx)

    # The first LOOKBACK points have no features, so original index i corresponds to row (i - LOOKBACK).
    train_end = s_train.stop - LOOKBACK
    val_end = s_val.stop - LOOKBACK
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    # Flat versions for tree / linear models.
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print(
        f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape} "
        f"(features per step={n_features})"
    )

    scores: list[Score] = []
    trained: dict[str, object] = {}
    # Per-model additive bias offset (in *scaled* space) applied after model.predict().
    bias_offsets: dict[str, float] = {}

    def inv_y(arr):
        return scaler_y.inverse_transform(np.asarray(arr).reshape(-1, 1)).ravel()

    # ---------------- Random Forest ----------------
    # Sklearn's RandomForestRegressor has no quantile loss, so we apply a post-hoc
    # bias correction: shift predictions by the QUANTILE-th percentile of validation
    # residuals, biasing toward over-prediction.
    print("\nTraining Random Forest with grid search...")
    rf_param_grid = {"n_estimators": [100, 200], "max_depth": [10, None]}
    best_rf_rmse = float("inf")
    best_rf_model = None
    best_rf_bias = 0.0

    for params in ParameterGrid(rf_param_grid):
        rf = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE, **params)
        rf.fit(X_train_flat, y_train)
        rf_val_residuals = y_val - rf.predict(X_val_flat)
        rf_bias = float(np.quantile(rf_val_residuals, QUANTILE))
        y_pred_val = rf.predict(X_val_flat) + rf_bias
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        if val_rmse < best_rf_rmse:
            best_rf_rmse = val_rmse
            best_rf_model = rf
            best_rf_bias = rf_bias

    rf = best_rf_model
    y_pred = rf.predict(X_test_flat) + best_rf_bias
    scores.append(evaluate("RandomForest", inv_y(y_test), inv_y(y_pred)))
    trained["RandomForest"] = rf
    bias_offsets["RandomForest"] = best_rf_bias

    # ---------------- XGBoost (quantile regression) ----------------
    try:
        from xgboost import XGBRegressor
        print("Training XGBoost with grid search...")
        xgb_param_grid = {
            "n_estimators": [200, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        }
        best_xgb_rmse = float("inf")
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
                **params,
            )
            xgb.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
            y_pred_val = xgb.predict(X_val_flat)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            if val_rmse < best_xgb_rmse:
                best_xgb_rmse = val_rmse
                best_xgb_model = xgb

        xgb = best_xgb_model
        y_pred = xgb.predict(X_test_flat)
        scores.append(evaluate("XGBoost", inv_y(y_test), inv_y(y_pred)))
        trained["XGBoost"] = xgb
        bias_offsets["XGBoost"] = 0.0
    except ImportError:
        print("xgboost not installed — skipping.")

    # ---------------- LightGBM (quantile regression) ----------------
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        print("Training LightGBM with grid search...")
        lgbm_param_grid = {
            "n_estimators": [500, 1000],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 63],
        }
        best_lgbm_rmse = float("inf")
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
                **params,
            )
            lgbm.fit(
                X_train_flat, y_train,
                eval_set=[(X_val_flat, y_val)],
                callbacks=[early_stopping(20), log_evaluation(0)],
            )
            y_pred_val = lgbm.predict(X_val_flat)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            if val_rmse < best_lgbm_rmse:
                best_lgbm_rmse = val_rmse
                best_lgbm_model = lgbm

        lgbm = best_lgbm_model
        y_pred = lgbm.predict(X_test_flat)
        scores.append(evaluate("LightGBM", inv_y(y_test), inv_y(y_pred)))
        trained["LightGBM"] = lgbm
        bias_offsets["LightGBM"] = 0.0
    except ImportError:
        print("lightgbm not installed — skipping.")

    # ---------------- LSTM (pinball loss) ----------------
    try:
        lstm = train_lstm(X_train, y_train, X_val, y_val, n_features)
        y_pred = predict_rnn(lstm, X_test)
        scores.append(evaluate("LSTM", inv_y(y_test), inv_y(y_pred)))
        trained["LSTM"] = lstm
        bias_offsets["LSTM"] = 0.0
    except ImportError:
        print("torch not installed — skipping LSTM.")

    # ---------------- GRU (pinball loss) ----------------
    try:
        gru = train_gru(X_train, y_train, X_val, y_val, n_features)
        y_pred = predict_rnn(gru, X_test)
        scores.append(evaluate("GRU", inv_y(y_test), inv_y(y_pred)))
        trained["GRU"] = gru
        bias_offsets["GRU"] = 0.0
    except ImportError:
        print("torch not installed — skipping GRU.")

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
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "target_idx": target_idx,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "quantile": QUANTILE,
        "bias_offset": bias_offsets.get(best.name, 0.0),
        "score": asdict(best),
        # Inference contract:
        #   1. take the last `lookback` rows of FEATURE_COLS as shape (lookback, n_features).
        #   2. scale with bundle["scaler_X"].
        #   3. for tree models: reshape to (1, lookback*n_features) and call model.predict(...).
        #      for LSTM/GRU: load rnn_path (.pth), feed shape (1, lookback, n_features).
        #   4. add bundle["bias_offset"] (scaled-space additive correction; 0 for quantile-trained models).
        #   5. inverse-transform the corrected scalar with bundle["scaler_y"].
    }

    if best.name in ("LSTM", "GRU"):
        import torch
        rnn_filename = "lstm.pth" if best.name == "LSTM" else "gru.pth"
        torch.save(trained[best.name].state_dict(), OUT_DIR / rnn_filename)
        bundle["rnn_path"] = rnn_filename
        bundle["n_features"] = n_features
    else:
        bundle["model"] = trained[best.name]

    bundle_path = OUT_DIR / "bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"Saved bundle -> {bundle_path}")

    metadata = {
        "best_model": best.name,
        "lookback": LOOKBACK,
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "csv": CSV_PATH.name,
        "n_samples": int(n),
        "quantile": QUANTILE,
        "metrics": {s.name: asdict(s) for s in scores},
        "files": (
            ["bundle.joblib", "lstm.pth"] if best.name == "LSTM"
            else ["bundle.joblib", "gru.pth"] if best.name == "GRU"
            else ["bundle.joblib"]
        ),
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata -> {OUT_DIR / 'metadata.json'}")


# ----------------------------- inference helper ------------------------------
def load_predictor(bundle_dir: str | Path):
    """
    Returns a callable predict(last_window) -> float, plus the loaded bundle.
    `last_window` must be a 2-D array of shape (lookback, n_features) whose
    columns are in the order given by bundle['feature_names'].
    """
    bundle_dir = Path(bundle_dir)
    bundle = joblib.load(bundle_dir / "bundle.joblib")
    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]
    lookback = bundle["lookback"]
    feature_names = bundle["feature_names"]
    n_features = len(feature_names)
    bias_offset = float(bundle.get("bias_offset", 0.0))

    if bundle["model_type"] in ("LSTM", "GRU"):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_fn = get_lstm_model if bundle["model_type"] == "LSTM" else get_gru_model
        rnn_model = model_fn(n_features).to(device)
        # support both old bundles (lstm_path) and new ones (rnn_path)
        rnn_file = bundle.get("rnn_path") or bundle.get("lstm_path")
        rnn_model.load_state_dict(torch.load(bundle_dir / rnn_file, map_location=device))
        rnn_model.eval()

        def predict(window):
            arr = np.asarray(window, dtype=float).reshape(lookback, n_features)
            scaled = scaler_X.transform(arr).reshape(1, lookback, n_features)
            with torch.no_grad():
                scaled_t = torch.tensor(scaled, dtype=torch.float32).to(device)
                scaled_pred = rnn_model(scaled_t).item() + bias_offset
            return float(scaler_y.inverse_transform([[scaled_pred]])[0, 0])
    else:
        model = bundle["model"]
        booster = getattr(model, "booster_", None)

        def predict(window):
            arr = np.asarray(window, dtype=float).reshape(lookback, n_features)
            scaled = scaler_X.transform(arr).reshape(1, lookback * n_features)
            raw = booster.predict(scaled) if booster is not None else model.predict(scaled)
            scaled_pred = float(np.ravel(raw)[0]) + bias_offset
            return float(scaler_y.inverse_transform([[scaled_pred]])[0, 0])

    return predict, bundle


if __name__ == "__main__":
    main()
    # Smoke-test the saved bundle:
    predict, meta = load_predictor(OUT_DIR)
    df = load_frame(CSV_PATH)
    last = df.tail(meta["lookback"]).to_numpy(dtype=float)
    print(
        f"\nSmoke test — next-minute {meta['target_name']} prediction "
        f"using last {meta['lookback']} samples: {predict(last):.2f}"
    )
