import time
import gymnasium as gym
import numpy as np
import torch
import gomoku_gym # Note if this doesn't work, run pip install -e . first
from gomoku_gym.MCTS import MCTS
from gomoku_gym.mcts_network import NetworkMCTS
from gomoku_gym.networks import GomokuNet


env = gym.make("gomoku_gym/GridWorld-v0", render_mode="human") # print board
# env = gym.make("gomoku_gym/GridWorld-v0") # dont print board
def MCTS_play():
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
            
def human_move():
    print("Enter move:")
    x = int(input("x: "))
    y = int(input("y: "))
    return x, y

def agent_move(mcts, prev_move):
    start = time.time()

    if prev_move:
        mcts.move(prev_move, env)
    # Get MCTS policy
    root_node = mcts.root
    total_visits = sum(child.N for child in root_node.children.values())

    best_move = mcts.search_parallel(num_simulations=80) # 16 cores

    mcts.move(best_move, env)

    print(f"Thought for {time.time() - start:.2f} seconds, {total_visits} total visits")
    return best_move

def playp1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board_size = env.unwrapped.board_size
    network = GomokuNet(board_size).to(device)
    network.load_state_dict(torch.load("gomoku_net_4x4_470-best.pth"))

    observation, info = env.reset()

    # agent is player 2
    mcts = NetworkMCTS(env, 2, network, board_size, 0)
    terminated = False
    while not terminated:
        valid = False
        while not valid:
            hx, hy = human_move()
            observation, reward, terminated, truncated, info = env.step(np.array([hx, hy]))
            valid = info.get("valid") # checks if the player made a valid move
            if not valid:
                print("Invalid move. Try again")
        if terminated:
            break

        valid = False
        while not valid:
            ax, ay = agent_move(mcts, (hx,hy))
            observation, reward, terminated, truncated, info = env.step(np.array([ax, ay]))
            valid = info.get("valid") # checks if the agent made a valid move
            if not valid:
                print("Invalid move. Error in AI Agent.")
        if terminated:
            break

def playp2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    board_size = env.unwrapped.board_size
    network = GomokuNet(board_size).to(device)
    network.load_state_dict(torch.load("gomoku_net_4x4_470-best.pth"))

    observation, info = env.reset()

    # agent is player 1
    mcts = NetworkMCTS(env, 1, network, board_size, 0)
    terminated = False
    hx = None
    hy = None
    while not terminated:
        valid = False
        while not valid:
            ax, ay = agent_move(mcts, (hx,hy))
            observation, reward, terminated, truncated, info = env.step(np.array([ax, ay]))
            valid = info.get("valid") # checks if the agent made a valid move
            if not valid:
                print("Invalid move. Error in AI Agent.")
        if terminated:
            break

        valid = False
        while not valid:
            hx, hy = human_move()
            observation, reward, terminated, truncated, info = env.step(np.array([hx, hy]))
            valid = info.get("valid") # checks if the player made a valid move
            if not valid:
                print("Invalid move. Try again")
        if terminated:
            break

playp1()

env.close()
