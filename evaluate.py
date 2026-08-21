import pygame
import numpy as np
from sb3_contrib import RecurrentPPO
from envs.maze_env import AntMazeEnv
import time

# --- Visual Settings ---
CELL_SIZE = 40
WHITE = (245, 245, 245)
BLACK = (40, 40, 40)
GREEN = (50, 200, 50)
BROWN = (139, 69, 19)
RED = (220, 20, 60)
GRID_LINE = (200, 200, 200)


def main():
    print("Loading Phase 1 LSTM Baseline...")
    env = AntMazeEnv(width=11, height=11, obs_window=3, min_food_dist=3)
    model = RecurrentPPO.load("ppo_ant_lstm_baseline")

    pygame.init()
    screen = pygame.display.set_mode((env.width * CELL_SIZE, env.height * CELL_SIZE + 40))
    pygame.display.set_caption("EAC Phase 1: LSTM Foraging Evaluation")
    font = pygame.font.SysFont(None, 24)

    obs, _ = env.reset()
    terminated, truncated = False, False
    running = True

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    print("Starting visual simulation...")
    total_reward = 0.0

    while running and not (terminated or truncated):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward

        episode_starts = np.zeros((1,), dtype=bool)

        # 1. Render Grid
        screen.fill(WHITE)
        for y in range(env.height):
            for x in range(env.width):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if env.grid[x, y] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                elif (x, y) == env.nest_pos:
                    pygame.draw.rect(screen, BROWN, rect)
                elif (x, y) == env.food_pos:
                    pygame.draw.rect(screen, GREEN, rect)
                else:
                    pygame.draw.rect(screen, GRID_LINE, rect, 1)

        # 2. Render Ant
        ant_x, ant_y = env.agent_pos
        center_x = ant_x * CELL_SIZE + CELL_SIZE // 2
        center_y = ant_y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, RED, (center_x, center_y), CELL_SIZE // 3)

        # 3. Render Status Bar
        status_text = f"Step: {env.step_count} | Reward: {total_reward:.2f}"
        text_surface = font.render(status_text, True, BLACK)
        screen.blit(text_surface, (10, env.height * CELL_SIZE + 10))

        pygame.display.flip()
        time.sleep(0.08)

    if terminated:
        print(f"Success! Reached food in {env.step_count} steps.")
    else:
        print("Timeout reached.")

    time.sleep(2)
    pygame.quit()


if __name__ == "__main__":
    main()