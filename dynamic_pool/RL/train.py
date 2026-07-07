"""Train an RL agent to choose container idle-timeout in the dynamic-pool sim.

Arrivals come from the real per-minute trace (arrival_data/ML_tests/sample.csv).
Each episode is a random ``max_steps * step_duration``-second window of the
trace, so the agent sees different traffic regimes across episodes.

The algorithm is selectable via the ``ALGORITHM`` knob below ("PPO" or "SAC").
Both work directly on the env's continuous Box action space — no env changes
are needed to switch between them.

Run:  python -m dynamic_pool.RL.train
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from variables import config as base_config
from dynamic_pool.RL.rl_env import DynamicPoolEnv


# ----- knobs -----
ALGORITHM = "PPO"          # "PPO" (on-policy) or "SAC" (off-policy)
TRACE_NAME = "non_station.csv"   # "non_station.csv" or "day_night.csv"
STEP_DURATION = 60.0
MAX_STEPS = 200            # 200 steps * 60s = 12000s = 200 trace-minutes per episode
IDLE_MIN = 1.0
IDLE_MAX = 60.0
RAM_PENALTY = 1.0
TOTAL_TIMESTEPS = 200_000
# Evaluate on a held-out window every EVAL_FREQ env-steps; the best-scoring
# policy (+ its VecNormalize stats) is checkpointed to <run_name>_best/.
EVAL_FREQ = 10_000
# Parallel rollout workers for PPO (on-policy → near-linear speedup since the
# SimPy step is CPU-bound). Set to ~number of physical cores. SAC ignores this
# (off-policy, runs single-env).
N_ENVS = 8
LOG_DIR = os.path.join(_HERE, "logs")

_TRACE_CONFIG = {
    "non_station.csv": {"log_subdir": "non_station"},
    "day_night.csv":   {"log_subdir": "day_night"},
}

if TRACE_NAME not in _TRACE_CONFIG:
    raise ValueError(
        f"Unknown TRACE_NAME {TRACE_NAME!r}; "
        f"expected one of {sorted(_TRACE_CONFIG)}."
    )

_TRACE_STEM = TRACE_NAME.replace(".csv", "")
TRACE_CSV = _REPO_ROOT / "arrival_data" / "ML_tests" / TRACE_NAME

# Per-algorithm hyperparameters. Anything not listed uses SB3 defaults.
ALGO_KWARGS = {
    "SAC": dict(
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        learning_starts=1_000,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        gamma=0.99,
    ),
    "PPO": dict(
        learning_rate=3e-4,
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
    ),
}
ALGO_CLASSES = {"PPO": PPO, "SAC": SAC}


def load_trace(csv_path: Path) -> np.ndarray:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                m = int(row["minute"])
                v = float(row["count"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append((m, v))
    rows.sort(key=lambda x: x[0])
    return np.asarray([v for _, v in rows], dtype=float)


class SimMetricsCallback(BaseCallback):
    """Log per-step simulator metrics from env info dict to TensorBoard."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "idle_timeout" not in info:
                continue
            self.logger.record_mean("sim/idle_timeout", float(info["idle_timeout"]))
            self.logger.record_mean("sim/cold_ratio", float(info["cold_ratio"]))
            self.logger.record_mean("sim/ram_util", float(info["ram_util"]))
            self.logger.record_mean("sim/d_gen", float(info["d_gen"]))
            self.logger.record_mean("sim/d_cold", float(info["d_cold"]))
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Fired by EvalCallback on each new best model — saves the training
    env's VecNormalize stats next to best_model.zip so the best checkpoint
    keeps a matching normalizer."""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None:
            vec_normalize.save(self.save_path)
            if self.verbose:
                print(f"  New best — VecNormalize stats saved to {self.save_path}")
        return True


def make_env(trace: np.ndarray, seed: int = 0, random_window: bool = True):
    def _init():
        env = DynamicPoolEnv(
            base_config=base_config,
            step_duration=STEP_DURATION,
            max_steps=MAX_STEPS,
            idle_timeout_min=IDLE_MIN,
            idle_timeout_max=IDLE_MAX,
            ram_penalty=RAM_PENALTY,
            seed=seed,
            arrival_mode="trace",
            trace=trace,
            random_window=random_window,
        )
        return Monitor(env)
    return _init


def build_vec_env(algo: str, trace: np.ndarray):
    """SubprocVecEnv with N_ENVS parallel workers for PPO; single-env
    DummyVecEnv for SAC. Each PPO worker gets a distinct seed to decorrelate
    rollouts. Wrapped in VecNormalize so the heterogeneous obs components
    (raw counts vs the [0,1] RAM ratio) and the rewards are standardized
    online — without it the policy is dominated by the large-count dims."""
    if algo == "PPO" and N_ENVS > 1:
        venv = SubprocVecEnv([make_env(trace, seed=i) for i in range(N_ENVS)])
    else:
        venv = DummyVecEnv([make_env(trace, seed=0)])
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


def build_model(algorithm: str, vec_env):
    """Construct the SB3 model for the requested algorithm."""
    algo = algorithm.upper()
    if algo not in ALGO_CLASSES:
        raise ValueError(
            f"Unknown ALGORITHM {algorithm!r}; choose from {sorted(ALGO_CLASSES)}."
        )
    return ALGO_CLASSES[algo](
        "MlpPolicy",
        vec_env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        **ALGO_KWARGS[algo],
    )


def main():
    algo = ALGORITHM.upper()
    if algo not in ALGO_CLASSES:
        raise ValueError(
            f"Unknown ALGORITHM {ALGORITHM!r}; choose from {sorted(ALGO_CLASSES)}."
        )
    run_name = f"{algo.lower()}_idle_timeout_{_TRACE_STEM}"
    best_dir = os.path.join(_HERE, f"{run_name}_best")
    best_model_path = os.path.join(best_dir, "best_model.zip")
    best_vecnorm_path = os.path.join(best_dir, "best_vecnorm.pkl")

    trace = load_trace(TRACE_CSV)
    n_envs = N_ENVS if (algo == "PPO" and N_ENVS > 1) else 1
    print(f"Algorithm: {algo}  (parallel envs: {n_envs})")
    print(f"Loaded trace: {len(trace)} minutes, "
          f"mean={trace.mean():.2f}, min={trace.min():.0f}, max={trace.max():.0f}")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    vec_env = build_vec_env(algo, trace)

    model = build_model(algo, vec_env)

    # Held-out evaluation on a fixed window. Whenever the mean eval reward
    # improves, EvalCallback writes best_dir/best_model.zip and the new-best
    # hook saves the matching VecNormalize stats. eval_freq counts callback
    # calls, so divide by n_envs to keep EVAL_FREQ in env-steps.
    eval_env = DummyVecEnv([make_env(trace, seed=999_999, random_window=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=best_dir,
        eval_freq=max(EVAL_FREQ // n_envs, 1),
        n_eval_episodes=1,
        deterministic=True,
        callback_on_new_best=SaveVecNormalizeCallback(best_vecnorm_path, verbose=1),
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        tb_log_name=run_name,
        callback=[SimMetricsCallback(), eval_callback],
    )
    print(f"Best model + VecNormalize saved in {best_dir}")

    # Quick deterministic rollout on a fixed window using the best checkpoint.
    # Reload the best-run VecNormalize stats with updates frozen so the eval
    # normalizes obs exactly the way inference will.
    if not os.path.exists(best_model_path):
        print("No best checkpoint was saved (eval never improved); "
              "skipping sanity-check rollout.")
        return
    eval_venv = DummyVecEnv([make_env(trace, seed=12345, random_window=False)])
    eval_venv = VecNormalize.load(best_vecnorm_path, eval_venv)
    eval_venv.training = False
    eval_venv.norm_reward = False
    best_model = ALGO_CLASSES[algo].load(best_model_path, env=eval_venv)
    obs = eval_venv.reset()
    total_r = 0.0
    for _ in range(MAX_STEPS):
        action, _ = best_model.predict(obs, deterministic=True)
        obs, r, done, _ = eval_venv.step(action)
        total_r += float(r[0])
        if done[0]:
            break
    print(f"Eval episode total reward (best model): {total_r:.3f}")


if __name__ == "__main__":
    main()
