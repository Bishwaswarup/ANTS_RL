import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from envs.maze_env import AntMazeEnv

def main():
    env = AntMazeEnv(width=11, height=11, obs_window=3, min_food_dist=3)

    print("Checking environment API compliance...")
    check_env(env, warn=True)
    print("Environment check passed!")

    print("Initializing Recurrent PPO Agent (LSTM) with exploration tuning...")
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        ent_coef=0.03,  # Entropy coefficient to maintain exploration
        tensorboard_log="./ppo_ant_tensorboard/"
    )

    print("Starting Phase 1 training with Oscillation Fix...")
    model.learn(total_timesteps=80000, progress_bar=True)

    model.save("ppo_ant_lstm_baseline")
    print("Training complete. Colony baseline saved.")

if __name__ == "__main__":
    main()