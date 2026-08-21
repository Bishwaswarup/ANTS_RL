"""
train.py — EAC Phase 1 Training
================================
Default strategy: PPO + Frame Stacking (no LSTM)
-------------------------------------------------
Why frame stacking instead of LSTM?

RecurrentPPO with LSTM is powerful but sensitive:
  - Requires long rollout sequences to propagate gradients through time
  - Prone to oscillation if the reward signal is noisy or poorly shaped
  - Slower to converge (~3× more wall-clock time for same timesteps)

Frame stacking gives the agent "memory" by concatenating the last N
observations as extra channels.  For maze navigation this is sufficient:
the agent can infer its direction of travel from how its local view shifted
between frames, without any recurrence.

  Stacked obs shape: (n_stack × 3, 7, 7) = (12, 7, 7) with n_stack=4

The CNN feature extractor (agents/policy.py) handles arbitrary channel
counts, so frame stacking drops in without any architecture changes.

To switch back to the LSTM approach, see the commented block at the bottom.
"""

import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from envs.maze_env import AntMazeEnv
from agents.policy import AntCNNExtractor


# ── Hyperparameters ──────────────────────────────────────────────────────────

N_ENVS          = 8          # Parallel environments — 8× faster data collection
N_STACK         = 4          # Frame stack depth (memory window)
TOTAL_STEPS     = 500_000    # Additional steps per run (stacks up across runs)
LEARNING_RATE   = 3e-4
N_STEPS         = 512        # Steps per env per update (rollout length)
BATCH_SIZE      = 256        # Mini-batch size for gradient updates
N_EPOCHS        = 10         # PPO epochs per update
ENT_COEF        = 0.05       # Entropy bonus — keeps exploration alive longer
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
FEATURES_DIM    = 64         # CNN embedding size

SAVE_PATH       = "ppo_ant_framstack_baseline"
TB_LOG          = "./ppo_ant_tensorboard/"
EVAL_FREQ       = 20_000     # Evaluate every N total steps
REWARD_THRESH   = 8.0        # Stop early if mean eval reward exceeds this


# ── Training ─────────────────────────────────────────────────────────────────

def make_env(min_food_dist: int = 3):
    """Factory for a single AntMazeEnv instance."""
    return AntMazeEnv(width=11, height=11, obs_window=3, min_food_dist=min_food_dist)


def main():
    # 1. Sanity-check the environment API (runs once, single env)
    print("Checking environment API compliance…")
    check_env(make_env(), warn=True)
    print("✓ Environment check passed\n")

    # 2. Vectorised + frame-stacked training environment
    print(f"Building {N_ENVS} parallel envs with {N_STACK}-frame stack…")
    train_env = make_vec_env(make_env, n_envs=N_ENVS)
    train_env = VecFrameStack(train_env, n_stack=N_STACK)

    # 3. Separate eval environment (single, unmodified)
    eval_env = VecFrameStack(make_vec_env(make_env, n_envs=1), n_stack=N_STACK)

    # 4. Callbacks
    reward_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESH, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./models/logs/",
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
        callback_on_new_best=reward_callback,
        verbose=1,
    )

    # 5. Load existing checkpoint OR build from scratch
    checkpoint = f"{SAVE_PATH}.zip"
    if Path(checkpoint).exists():
        print(f"▶ Resuming from checkpoint: {checkpoint}")
        model = PPO.load(
            SAVE_PATH,
            env=train_env,
            tensorboard_log=TB_LOG,
            # Keep all original hyperparams — only the weights are loaded
        )
        reset_num_timesteps = False   # continue the timestep counter
    else:
        print("▶ No checkpoint found — starting fresh training…")
        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            policy_kwargs=dict(
                features_extractor_class=AntCNNExtractor,
                features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
            ),
            tensorboard_log=TB_LOG,
        )
        reset_num_timesteps = True

    # 6. Train
    total_so_far = getattr(model, "num_timesteps", 0)
    print(f"Training for {TOTAL_STEPS:,} more steps  (total so far: {total_so_far:,})…")
    print("Monitor live:  tensorboard --logdir ppo_ant_tensorboard/\n")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    # 7. Save (overwrites previous checkpoint — best_model.zip is kept separately)
    model.save(SAVE_PATH)
    print(f"\n✓ Training complete. Model saved → {SAVE_PATH}.zip")
    print(f"✓ Best checkpoint  → models/best/best_model.zip")


# ── LSTM alternative (uncomment to use instead of frame-stacking) ─────────────
#
# from sb3_contrib import RecurrentPPO
#
# model = RecurrentPPO(
#     "MlpLstmPolicy",
#     train_env,            # use make_vec_env WITHOUT VecFrameStack
#     verbose=1,
#     learning_rate=3e-4,
#     n_steps=512,          # ← critical: default 128 is too short for maze memory
#     batch_size=64,        # must be < n_steps * n_envs
#     n_epochs=10,
#     ent_coef=0.05,
#     policy_kwargs=dict(
#         features_extractor_class=AntCNNExtractor,
#         features_extractor_kwargs=dict(features_dim=64),
#     ),
#     tensorboard_log=TB_LOG,
# )
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
