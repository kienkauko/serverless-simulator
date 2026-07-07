"""Plot static idle-timeout sweep results across arrival rates.

Reads ``logs/dynamic/arrival_<X>/<datetime>_static_sweep_x<Y>.csv`` produced by
``dynamic_pool/RL/infer.py`` when USE_ML=False.
  - X = arrival rate (parsed from the folder name)
  - Y = repetition index (1-based, parsed from the filename)

For each metric, draws one curve per arrival rate vs idle timeout, averaging
across repetitions with a 95% CI band.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent  # logs/dynamic/

# ----------------------------- style -----------------------------------------
LINE_WIDTH   = 3.5
MARKER_SIZE  = 10
LABEL_SIZE   = 20
TICK_SIZE    = 20
LEGEND_SIZE  = 19
LEGEND_TITLE_SIZE = 19
FIGURE_SIZE  = (7, 5)
CI_ALPHA     = 0.2

LOAD_LABEL = {
    2.35: "5%",
    4.7:  "10%",
    9.4:  "20%",
}

MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h"]
CMAP = plt.get_cmap("tab10")

# CSV columns (from dynamic_pool/RL/infer.py summarize_run):
#   blocking_prob, mean_latency, mean_cpu, mean_ram, ...
def _cpu_per_req(r):
    acc = r.get("accepted", 0)
    return (r["mean_cpu"] * r["sim_time"] / acc) if acc else float("nan")

def _ram_per_req(r):
    acc = r.get("accepted", 0)
    return (r["mean_ram"] * r["sim_time"] / acc) if acc else float("nan")

METRICS = [
    ("blocking_rate", "Blocking Ratio (%)",
     lambda r: r["blocking_prob"], 100.0),
    ("response_time", "Mean Processing Time (s)",
     lambda r: r["mean_latency"], 1.0),
    ("cpu", "Mean CPUsecond Usage",
     lambda r: r["mean_cpu"], 1.0),
    ("ram_pct", "Mean RAMsecond Usage",
     lambda r: r["mean_ram"], 1.0),
    ("cpu_per_req", "Mean CPUsecond per Request",
     _cpu_per_req, 1.0),
    ("ram_per_req", "Mean RAMsecond per Request",
     _ram_per_req, 1.0),
]

# Per-arrival-rate breakdown of time spent in each container state.
PHASE_METRICS = [
    ("cold_starting", "Cold starting", lambda r: r["mean_cold_starting"]),
    ("processing",    "Processing",    lambda r: r["mean_processing"]),
    ("idle",          "Idle",          lambda r: r["mean_idle"]),
]

ARRIVAL_DIR_RE = re.compile(r"^arrival_([\d.]+)$")
REP_FILE_RE = re.compile(r"_static_sweep_x(\d+)\.csv$")


# ----------------------------- data loading ----------------------------------
def _to_float(v):
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_data():
    """Return ``{arrival_rate: [rep_rows, ...]}``.

    Each ``rep_rows`` is a list of dict rows sorted by ``timeout``. Reps with
    a mismatching timeout grid are skipped with a warning.
    """
    data: dict[float, list[list[dict]]] = {}
    for sub in sorted(HERE.iterdir()):
        if not sub.is_dir():
            continue
        m = ARRIVAL_DIR_RE.match(sub.name)
        if not m:
            continue
        rate = float(m.group(1))

        rep_files = []
        for p in sub.glob("*_static_sweep_x*.csv"):
            rm = REP_FILE_RE.search(p.name)
            if rm:
                rep_files.append((int(rm.group(1)), p))
        rep_files.sort()

        reps = []
        for _, csv_path in rep_files:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                rows = [{k: _to_float(v) for k, v in r.items()}
                        for r in csv.DictReader(f)]
            rows.sort(key=lambda r: r["timeout"])
            reps.append(rows)

        if reps:
            data[rate] = reps
    return data


def build_curve(reps, getter, scale):
    """Compute (timeouts, mean, ci_lo, ci_hi) across reps for one metric."""
    timeouts = np.array([r["timeout"] for r in reps[0]], dtype=float)
    M = np.array([[getter(row) * scale for row in rep] for rep in reps],
                 dtype=float)
    mean = np.nanmean(M, axis=0)
    if M.shape[0] > 1:
        std = np.nanstd(M, axis=0, ddof=1)
        sem = std / np.sqrt(M.shape[0])
        ci = 1.96 * sem
    else:
        ci = np.zeros_like(mean)
    return timeouts, mean, mean - ci, mean + ci


# ----------------------------- plotting --------------------------------------
def main():
    data = load_data()
    if not data:
        raise SystemExit(f"No arrival_* folders with CSVs found under {HERE}")

    rates = sorted(data.keys())
    print(f"Found {len(rates)} arrival rate(s): {rates}")
    for rate in rates:
        print(f"  arrival={rate}: {len(data[rate])} repetition(s)")

    for name, ylabel, getter, scale in METRICS:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for i, rate in enumerate(rates):
            t, mean, lo, hi = build_curve(data[rate], getter, scale)
            label = LOAD_LABEL.get(rate, f"rate={rate}")
            color = CMAP(i % 10)
            marker = MARKERS[i % len(MARKERS)]
            ax.plot(t, mean,
                    marker=marker, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, color=color, label=label,
                    markevery=max(1, len(t) // 20))
            ax.fill_between(t, lo, hi, color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Idle Timeout (s)", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(title="Load", fontsize=LEGEND_SIZE,
                  title_fontsize=LEGEND_TITLE_SIZE)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    # One figure per arrival rate: cold_starting / processing / idle vs timeout.
    for i, rate in enumerate(rates):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        for j, (_, label, getter) in enumerate(PHASE_METRICS):
            t, mean, lo, hi = build_curve(data[rate], getter, 1.0)
            color = CMAP(j % 10)
            marker = MARKERS[j % len(MARKERS)]
            ax.plot(t, mean,
                    marker=marker, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, color=color, label=label,
                    markevery=max(1, len(t) // 20))
            ax.fill_between(t, lo, hi, color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Idle Timeout (s)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Mean Function Number", fontsize=LABEL_SIZE)
        # ax.set_title(f"Load {LOAD_LABEL.get(rate, rate)}", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(title="Phase", fontsize=LEGEND_SIZE,
                  title_fontsize=LEGEND_TITLE_SIZE)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
