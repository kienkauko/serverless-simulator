"""Batch runner: execute every simulator strategy over both arrival traces.

Launches each strategy as its own subprocess, driving the per-run knobs through
environment variables (SIM_TRACE_NAME / SIM_USE_ML / SIM_USE_RL / SIM_NUM_WORKERS)
that each script reads at import time. Env vars are used (rather than in-process
global patching) because the parallel scripts spawn worker processes that
re-import the module fresh — only the inherited environment survives that
re-import.

Cases executed (sequentially):
  * fixed_pool/multi_ML_proactive_SR.py — static warm pool (square-root staffing)
        traces: non_station, day_night
  * dynamic_pool/multi_dynamic.py — dynamic-pool idle-timeout control, all 3 modes:
        STATIC   (USE_ML=False)             — fixed idle timeout from variables.py
        ANALYSIS (USE_ML=True,  USE_RL=False) — analytical controller
        RL       (USE_ML=True,  USE_RL=True)  — trained PPO/SAC checkpoint
        each x traces: non_station, day_night

Each case writes its own combined CSV under logs/<trace>/ as usual. Failures are
reported but do not stop the batch.

Run:  python run_all_cases.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
NUM_WORKERS = "6"
TRACES = ["non_station.csv", "day_night.csv"]


def _case(label, argv, env_overrides):
    """Build one case: (label, subprocess argv, extra env vars)."""
    return {"label": label, "argv": argv, "env": env_overrides}


def build_cases():
    cases = []

    # --- static warm pool ---
    # for trace in TRACES:
    #     cases.append(_case(
    #         f"SR  [{trace}]",
    #         [PYTHON, "fixed_pool/multi_ML_proactive_SR.py"],
    #         {"SIM_TRACE_NAME": trace, "SIM_NUM_WORKERS": NUM_WORKERS},
    #     ))

    # --- dynamic_pool: all three control modes x both traces ---
    # (mode label, SIM_USE_ML, SIM_USE_RL)
    analysis_modes = [
        # ("static",   "False", "False"),
        ("analysis", "True",  "False"),
        ("rl",       "True",  "True"),
    ]
    for mode_label, use_ml, use_rl in analysis_modes:
        for trace in TRACES:
            cases.append(_case(
                f"{mode_label}  [{trace}]",
                [PYTHON, "dynamic_pool/multi_dynamic.py"],
                {"SIM_TRACE_NAME": trace,
                 "SIM_USE_ML": use_ml,
                 "SIM_USE_RL": use_rl,
                 "SIM_NUM_WORKERS": NUM_WORKERS},
            ))

    return cases


def run_case(case):
    env = os.environ.copy()
    env.update(case["env"])
    env_str = "  ".join(f"{k}={v}" for k, v in case["env"].items())
    print("\n" + "=" * 78)
    print(f">>> {case['label']}")
    print(f"    cmd: {' '.join(case['argv'])}")
    print(f"    env: {env_str}")
    print("=" * 78, flush=True)

    start = datetime.now()
    result = subprocess.run(case["argv"], cwd=str(HERE), env=env)
    elapsed = (datetime.now() - start).total_seconds()
    ok = result.returncode == 0
    print(f"<<< {case['label']}  ->  "
          f"{'OK' if ok else f'FAILED (exit {result.returncode})'}  "
          f"({elapsed:.0f}s)", flush=True)
    return ok


def main():
    cases = build_cases()
    print(f"Running {len(cases)} cases (NUM_WORKERS={NUM_WORKERS} where applicable).")
    results = []
    for case in cases:
        results.append((case["label"], run_case(case)))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for label, ok in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {label}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} cases succeeded.")
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
