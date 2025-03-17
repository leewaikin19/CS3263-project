import gymnasium as gym
import numpy as np
import gomoku_gym # Note if this doesn't work, run pip install -e . first


#env = gym.make("gomoku_gym/GridWorld-v0", render_mode="human") # print board
env = gym.make("gomoku_gym/GridWorld-v0") # dont print board

observation, info = env.reset()
episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

    # play against human 
    """
    if info["valid"]:
        valid = False
        while not valid:
            x = int(input())
            y = int(input())
            observation, reward, terminated, truncated, info = env.step(np.array([x, y]))
            valid = info["valid"]"
            """

env.close()
