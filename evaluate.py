"""
evaluate.py — EAC Phase 1 Visual Evaluation
============================================
Loads the trained PPO (frame-stack) model and runs a live Pygame visualisation.
The agent's trail is shown in light red so you can see its exploration path.

Usage
-----
    python evaluate.py               # frame-stack PPO (default)
    python evaluate.py --lstm        # old LSTM baseline
    python evaluate.py --episodes 5  # run 5 episodes
"""

import sys
import argparse
import numpy as np
import pygame
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from envs.maze_env import AntMazeEnv

# ── Visual settings ──────────────────────────────────────────────────────────
CELL_SIZE = 40
WHITE     = (245, 245, 245)
BLACK     = (40,  40,  40)
GREEN     = (50,  200, 50)
BROWN     = (139, 69,  19)
RED       = (220, 20,  60)
GRID_LINE = (200, 200, 200)
TRAIL     = (255, 200, 200)   # light red — visited cells
N_STACK   = 4                 # must match train.py


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(use_lstm: bool):
    if use_lstm:
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError:
            print("sb3-contrib not installed. Run: pip install sb3-contrib")
            sys.exit(1)
        print("Loading LSTM baseline (ppo_ant_lstm_baseline)…")
        return RecurrentPPO.load("ppo_ant_lstm_baseline"), "lstm"

    for path in ("models/best/best_model", "ppo_ant_framstack_baseline"):
        try:
            model = PPO.load(path)
            print(f"Loaded model → {path}.zip")
            return model, "frame_stack"
        except FileNotFoundError:
            continue
    print(
        "No trained model found. Run train.py first.\n"
        "Expected: ppo_ant_framstack_baseline.zip  or  models/best/best_model.zip"
    )
    sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm",     action="store_true", help="Use LSTM baseline")
    parser.add_argument("--episodes", type=int,   default=1,    help="Episodes to run")
    parser.add_argument("--delay",    type=float, default=0.08, help="Seconds per frame")
    args = parser.parse_args()

    model, model_type = load_model(args.lstm)

    # Raw env for rendering (mirrors the vec_env's state)
    render_env = AntMazeEnv(width=11, height=11, obs_window=3, min_food_dist=3)

    # Inference env
    if model_type == "frame_stack":
        vec_env = VecFrameStack(
            DummyVecEnv([lambda: AntMazeEnv(width=11, height=11, obs_window=3, min_food_dist=3)]),
            n_stack=N_STACK
        )

    pygame.init()
    W = render_env.width  * CELL_SIZE
    H = render_env.height * CELL_SIZE + 50
    screen = pygame.display.set_mode((W, H))
    title  = "EAC — Frame-Stack PPO" if model_type == "frame_stack" else "EAC — LSTM Baseline"
    pygame.display.set_caption(title)
    font    = pygame.font.SysFont(None, 24)
    font_sm = pygame.font.SysFont(None, 18)

    for ep in range(args.episodes):
        # Reset both envs together
        render_obs, _ = render_env.reset()
        if model_type == "frame_stack":
            obs = vec_env.reset()
            # Align render_env grid/positions with vec_env (reset separately; maze differs)
        else:
            obs = render_obs
            lstm_states    = None
            episode_starts = np.ones((1,), dtype=bool)

        terminated, truncated = False, False
        running      = True
        total_reward = 0.0
        visited      = set()
        visited.add(tuple(render_env.agent_pos))

        print(f"\n── Episode {ep + 1} / {args.episodes} ──")

        while running and not (terminated or truncated):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Predict action
            if model_type == "frame_stack":
                action, _ = model.predict(obs, deterministic=True)
                obs, reward_arr, done_arr, _ = vec_env.step(action)
                reward     = float(reward_arr[0])
                terminated = bool(done_arr[0])
                truncated  = False
                # Mirror step in render_env so we can draw its state
                render_env.step(int(action[0]))
            else:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_starts, deterministic=True
                )
                obs, reward, terminated, truncated, _ = render_env.step(action.item())
                episode_starts = np.zeros((1,), dtype=bool)

            total_reward += reward
            visited.add(tuple(render_env.agent_pos))

            # ── Draw ────────────────────────────────────────────────────────
            screen.fill(WHITE)

            for y in range(render_env.height):
                for x in range(render_env.width):
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    if render_env.grid[x, y] == 1:
                        pygame.draw.rect(screen, BLACK, rect)
                    elif (x, y) == render_env.nest_pos:
                        pygame.draw.rect(screen, BROWN, rect)
                        screen.blit(font_sm.render("N", True, WHITE),
                                    (rect.centerx - 5, rect.centery - 6))
                    elif (x, y) == render_env.food_pos:
                        pygame.draw.rect(screen, GREEN, rect)
                        screen.blit(font_sm.render("F", True, WHITE),
                                    (rect.centerx - 5, rect.centery - 6))
                    elif (x, y) in visited:
                        pygame.draw.rect(screen, TRAIL, rect)
                        pygame.draw.rect(screen, GRID_LINE, rect, 1)
                    else:
                        pygame.draw.rect(screen, GRID_LINE, rect, 1)

            # Ant
            ax, ay = render_env.agent_pos
            cx = ax * CELL_SIZE + CELL_SIZE // 2
            cy = ay * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, RED, (cx, cy), CELL_SIZE // 3)

            # Status bar
            bar_y = render_env.height * CELL_SIZE
            pygame.draw.rect(screen, (230, 230, 230), pygame.Rect(0, bar_y, W, 50))
            status = (f"Ep {ep+1}  |  Step: {render_env.step_count}"
                      f"  |  Reward: {total_reward:.2f}"
                      f"  |  Visited: {len(visited)} cells"
                      f"  |  {'[Frame-Stack PPO]' if model_type == 'frame_stack' else '[LSTM]'}")
            screen.blit(font.render(status, True, BLACK), (10, bar_y + 15))
            pygame.display.flip()
            time.sleep(args.delay)

        outcome = f"✓ SUCCESS in {render_env.step_count} steps" if terminated else "✗ TIMEOUT"
        print(f"  {outcome} | reward {total_reward:.2f} | {len(visited)} unique cells")
        time.sleep(1.5)

    pygame.quit()


if __name__ == "__main__":
    main()