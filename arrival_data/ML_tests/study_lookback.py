"""
Study how prediction accuracy changes with the lookback window.

Methodology (standard train / validation / test protocol):
  * The series is split chronologically into 70% train, 15% validation, 15% test
    (same split as train_predictors.py).
  * For each lookback we train EVERY available algorithm on the train set and
    measure its error on the VALIDATION set.
  * Model/lookback selection is done on VALIDATION only (no peeking at test).
  * The winning (lookback, algorithm) combo is then reported on the held-out
    TEST set as the final, unbiased accuracy estimate.

Algorithms (mirrors train_predictors.py, single fast config each — no grid
search, since the goal here is comparing lookbacks, not squeezing each model):
  Linear, RandomForest, XGBoost, LightGBM, LSTM.

The "best lookback" is the one whose best algorithm has the lowest validation
RMSE. Lower MAE/RMSE/MAPE is better; higher R^2 is better.

Run:  python study_lookback.py
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Reuse the exact data-handling helpers so the study matches training.
from train_predictors import (
    CSV_PATH,
    TRAIN_FRAC,
    VAL_FRAC,
    RANDOM_STATE,
    load_series,
    chrono_split,
    make_supervised,
    train_lstm,
    predict_lstm,
)

warnings.filterwarnings("ignore")

LOOKBACKS = [5, 10, 15, 30, 60, 90, 120, 150, 180]
INCLUDE_LSTM = True  # set False to skip the (slow) LSTM across all lookbacks


# ----------------------------- metrics ---------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


# ----------------------------- algorithms ------------------------------------
def get_sklearn_models():
    """Return {name: factory()} for every installed tabular algorithm."""
    models = {}

    from sklearn.linear_model import LinearRegression
    models["Linear"] = lambda: LinearRegression()

    from sklearn.ensemble import RandomForestRegressor
    models["RandomForest"] = lambda: RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )
    except ImportError:
        print("  (xgboost not installed — skipping)")

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        print("  (lightgbm not installed — skipping)")

    return models


# ----------------------------- per-lookback eval -----------------------------
def evaluate_lookback(series: np.ndarray, lookback: int) -> dict:
    """Train every algorithm at this lookback; return {algo: {'val':..,'test':..}}."""
    n = len(series)
    s_train, s_val, s_test = chrono_split(n, TRAIN_FRAC, VAL_FRAC)

    # Scale on the training portion only (no leakage from val/test).
    scaler = MinMaxScaler()
    scaler.fit(series[s_train].reshape(-1, 1))
    scaled = scaler.transform(series.reshape(-1, 1)).ravel()

    X_all, y_all = make_supervised(scaled, lookback)
    # Row i of X corresponds to original series index (i + lookback).
    train_end = s_train.stop - lookback
    val_end = s_val.stop - lookback
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]

    inv = lambda a: scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
    yt_val, yt_test = inv(y_val), inv(y_test)

    out: dict[str, dict] = {}

    # ---- tabular algorithms ----
    for name, factory in get_sklearn_models().items():
        model = factory()
        model.fit(X_train, y_train)
        out[name] = {
            "val": metrics(yt_val, inv(model.predict(X_val))),
            "test": metrics(yt_test, inv(model.predict(X_test))),
        }

    # ---- LSTM (reuses train_predictors helpers) ----
    if INCLUDE_LSTM:
        try:
            lstm = train_lstm(X_train, y_train, X_val, y_val, lookback, scaler)
            out["LSTM"] = {
                "val": metrics(yt_val, inv(predict_lstm(lstm, X_val, lookback))),
                "test": metrics(yt_test, inv(predict_lstm(lstm, X_test, lookback))),
            }
        except ImportError:
            print("  (torch not installed — skipping LSTM)")

    return out


# ----------------------------- reporting -------------------------------------
def print_matrix(title: str, results: dict, algos: list[str], split: str, metric: str):
    print(f"\n=== {title} ===")
    header = f"{'Lookback':>9} " + " ".join(f"{a:>12}" for a in algos)
    print(header)
    print("-" * len(header))
    for lb in LOOKBACKS:
        row = f"{lb:>9} "
        row += " ".join(f"{results[lb][a][split][metric]:>12.4f}"
                         if a in results[lb] else f"{'-':>12}" for a in algos)
        print(row)


def save_full_report(path: Path, series, results, algos, per_lookback_best,
                     best_lb, best_algo):
    """Dump every algorithm's validation + test metrics at every lookback."""
    lines = []
    lines.append("Lookback-window study — full results")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Data file: {CSV_PATH.name}  |  {len(series)} samples  |  "
                 f"range=[{series.min():.0f}, {series.max():.0f}]  mean={series.mean():.2f}")
    lines.append(f"Split (chronological): {TRAIN_FRAC:.0%} train / {VAL_FRAC:.0%} val / "
                 f"{1 - TRAIN_FRAC - VAL_FRAC:.0%} test")
    lines.append(f"Lookbacks tested: {LOOKBACKS}")
    lines.append(f"Algorithms: {', '.join(algos)}")
    lines.append("Selection rule: lowest VALIDATION RMSE; TEST reported as unbiased estimate.")
    lines.append("")

    # Per-(lookback, algorithm) full metric dump.
    for split in ("val", "test"):
        label = "VALIDATION" if split == "val" else "TEST"
        lines.append("=" * 90)
        lines.append(f"{label} METRICS — all algorithms, all lookbacks")
        lines.append("=" * 90)
        header = (f"{'Lookback':>9} {'Algorithm':>14} {'MAE':>12} {'MSE':>14} "
                  f"{'RMSE':>12} {'R2':>10} {'MAPE(%)':>10}")
        lines.append(header)
        lines.append("-" * len(header))
        for lb in LOOKBACKS:
            for a in algos:
                if a not in results[lb]:
                    continue
                m = results[lb][a][split]
                lines.append(f"{lb:>9} {a:>14} {m['mae']:>12.4f} {m['mse']:>14.4f} "
                             f"{m['rmse']:>12.4f} {m['r2']:>10.4f} {m['mape']:>10.4f}")
            lines.append("-" * len(header))
        lines.append("")

    # Best algorithm per lookback (selected on validation).
    lines.append("=" * 90)
    lines.append("BEST ALGORITHM PER LOOKBACK (selected on validation RMSE)")
    lines.append("=" * 90)
    header = (f"{'Lookback':>9} {'BestAlgo':>14} {'ValRMSE':>10} {'TestRMSE':>10} "
              f"{'TestMAE':>10} {'TestR2':>10} {'TestMAPE':>10}")
    lines.append(header)
    lines.append("-" * len(header))
    for lb in LOOKBACKS:
        ba = per_lookback_best[lb]
        v = results[lb][ba]["val"]
        t = results[lb][ba]["test"]
        lines.append(f"{lb:>9} {ba:>14} {v['rmse']:>10.4f} {t['rmse']:>10.4f} "
                     f"{t['mae']:>10.4f} {t['r2']:>10.4f} {t['mape']:>10.4f}")
    lines.append("")

    # Overall winner.
    t = results[best_lb][best_algo]["test"]
    v = results[best_lb][best_algo]["val"]
    lines.append("=" * 90)
    lines.append(f"OVERALL WINNER: lookback={best_lb}, algorithm={best_algo}")
    lines.append(f"  validation: RMSE={v['rmse']:.4f}  MAE={v['mae']:.4f}  MSE={v['mse']:.4f}  "
                 f"R2={v['r2']:.4f}  MAPE={v['mape']:.4f}%")
    lines.append(f"  test:       RMSE={t['rmse']:.4f}  MAE={t['mae']:.4f}  MSE={t['mse']:.4f}  "
                 f"R2={t['r2']:.4f}  MAPE={t['mape']:.4f}%")
    lines.append("=" * 90)

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved full results -> {path}")


def main():
    series = load_series(CSV_PATH)
    print(f"Data: {CSV_PATH.name}  |  {len(series)} samples  |  "
          f"range=[{series.min():.0f}, {series.max():.0f}]  mean={series.mean():.1f}")
    print(f"Split: {TRAIN_FRAC:.0%} train / {VAL_FRAC:.0%} val / "
          f"{1 - TRAIN_FRAC - VAL_FRAC:.0%} test (chronological)")

    results: dict[int, dict] = {}
    algos: list[str] = []
    for lb in LOOKBACKS:
        print(f"\nLookback {lb}: training algorithms...")
        results[lb] = evaluate_lookback(series, lb)
        for a in results[lb]:
            if a not in algos:
                algos.append(a)
        done = ", ".join(results[lb].keys())
        print(f"  done: {done}")

    # Validation RMSE drives selection; test RMSE shown for reference only.
    print_matrix("VALIDATION RMSE (used for selection)", results, algos, "val", "rmse")
    print_matrix("TEST RMSE (held out, for reference)", results, algos, "test", "rmse")

    # ---- best algorithm per lookback (by validation RMSE) ----
    print("\n=== Best algorithm per lookback (selected on validation) ===")
    header = f"{'Lookback':>9} {'BestAlgo':>14} {'ValRMSE':>10} {'TestRMSE':>10} {'TestMAE':>10} {'TestMAPE':>10}"
    print(header)
    print("-" * len(header))
    per_lookback_best = {}
    for lb in LOOKBACKS:
        best_algo = min(results[lb], key=lambda a: results[lb][a]["val"]["rmse"])
        per_lookback_best[lb] = best_algo
        v = results[lb][best_algo]["val"]
        t = results[lb][best_algo]["test"]
        print(f"{lb:>9} {best_algo:>14} {v['rmse']:>10.4f} "
              f"{t['rmse']:>10.4f} {t['mae']:>10.4f} {t['mape']:>10.4f}")

    # ---- overall winner: lowest validation RMSE across all (lookback, algo) ----
    best_lb = min(LOOKBACKS, key=lambda lb: results[lb][per_lookback_best[lb]]["val"]["rmse"])
    best_algo = per_lookback_best[best_lb]
    v = results[best_lb][best_algo]["val"]
    t = results[best_lb][best_algo]["test"]

    print("\n" + "=" * 60)
    print(f"MOST PRECISE LOOKBACK: {best_lb}  (algorithm: {best_algo})")
    print(f"  selected on validation RMSE = {v['rmse']:.4f}")
    print(f"  final TEST metrics -> RMSE={t['rmse']:.4f}  MAE={t['mae']:.4f}  "
          f"R^2={t['r2']:.4f}  MAPE={t['mape']:.2f}%")
    print("=" * 60)

    # Save the full, detailed metric table for paper reporting.
    out_path = CSV_PATH.parent / f"lookback_study_{CSV_PATH.stem}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    save_full_report(out_path, series, results, algos, per_lookback_best,
                     best_lb, best_algo)


if __name__ == "__main__":
    main()
