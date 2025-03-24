import gymnasium as gym
import numpy as np
import gomoku_gym # Note if this doesn't work, run pip install -e . first
from gomoku_gym.MCTS import MCTS


env = gym.make("gomoku_gym/GridWorld-v0", render_mode="human") # print board
# env = gym.make("gomoku_gym/GridWorld-v0") # dont print board

observation, info = env.reset()
episode_over = False

MCTS_Agent = MCTS(env, 1)

while not episode_over:
    move = MCTS_Agent.search(num_simulations=100)
    observation, reward, terminated, truncated, info = env.step(np.array(move))
    MCTS_Agent.move(move, env)
    episode_over = terminated or truncated

    if not episode_over and info.get("valid", False):
        valid = False
        while not valid:
            print("Enter move:")
            x = int(input("x: "))
            y = int(input("y: "))
            observation, reward, terminated, truncated, info = env.step(np.array([x, y]))
            MCTS_Agent.move((x, y), env)
            valid = info.get("valid", False)
            if not valid:
                print("Invalid")
        episode_over = terminated or truncated
            

env.close()
