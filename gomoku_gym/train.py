import gymnasium as gym
import gomoku_gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from networks import GomokuNet
from mcts_network import NetworkMCTS
import gc
import time

class SelfPlayTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make("gomoku_gym/GridWorld-v0")
        self.board_size = self.env.unwrapped.board_size
        self.network = GomokuNet(self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001, weight_decay=0.01)
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.p1done = 0
        self.p2done = 0

        self.iter_cont = 0
        #self.iter_cont = 500
        #self.network.load_state_dict(torch.load(f"gomoku_net_4x4_{self.iter_cont}.pth"))
        #self.network.load_state_dict(torch.load(f"gomoku_net_4x4_470-best.pth"))

    def self_play(self, num_games):
        env = self.env
        gc.collect()
        np.set_printoptions(precision=2, suppress=True)

        for i in range(num_games):
            print("Game", i+1)

            observation, info = env.reset()
            episode_over = False
            history = []

            mcts = NetworkMCTS(env, self.env.unwrapped.player, self.network, self.board_size, enable_dirichlet=True)
            move_num = 1
            while not episode_over:
                start = time.time()
                # Get MCTS policy
                root_node = mcts.root

                #env.unwrapped._render_frame()

                #best_move = mcts.search(num_simulations=200)

                best_move = mcts.search_parallel(num_simulations=150) # 8 cores

                # calculate policy after the mcts search
                policy = np.zeros((self.board_size, self.board_size))
                total_visits = sum(child.N for child in root_node.children.values())

                #env.unwrapped._render_frame()

                for move, child in root_node.children.items():
                    x, y = move
                    policy[y][x] = child.N / total_visits

                observation, reward, terminated, truncated, info = env.step(np.array(best_move, dtype=np.int32))
                mcts.move(best_move, env)
                env.unwrapped._render_frame()
                episode_over = terminated or truncated
                print("Move", move_num, f"in {time.time() - start:.2f} seconds, {total_visits} total visits")
                #print("Policy", *policy)
                boards = self.network.board_to_tensor_symmetries(env.unwrapped._p1, env.unwrapped._p2, root_node.player, policy)
                for board_tensor, policy in boards:
                    #print(board_tensor)
                    #print(policy)
                    #print()
                    history.append((board_tensor, policy, root_node.player))
                move_num += 1

            #env.unwrapped._render_frame()
            # Determine final reward
            if env.unwrapped._win(env.unwrapped._p1):
                final_value = 1
            elif env.unwrapped._win(env.unwrapped._p2):
                # define p2 as negative reward, meaning nn is trained for p1
                final_value = -1
            else:
                final_value = 0

            #if (self.p1done < 5 and final_value == 1) or (self.p2done < 5 and final_value == -1):
                # Add to memory with appropriate value targets
            #    for (board_tensor, policy, player) in history:
            #        value_target = final_value # if player == 1 else -final_value
            #        self.memory.append((board_tensor, policy, value_target))
            for (board_tensor, policy, player) in history:
                value_target = final_value # if player == 1 else -final_value
                self.memory.append((board_tensor, policy, value_target))


            #if final_value == 1:
            #    self.p1done += 1
                #print("P1 done", self.p1done)
            #elif final_value == -1:
            #    self.p2done += 1
                #print("P2 done", self.p2done)

            env.close()
            #if self.p1done >= 5 and self.p2done >= 5:
            #    break
        #print("self", self.p1done, self.p2done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        boards, policies, values = zip(*batch)

        boards = torch.cat(boards).to(self.device)
        policies = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1).to(self.device)
        # Forward pass
        self.optimizer.zero_grad()
        log_probs, value_preds = self.network(boards)

        log_probs = log_probs.reshape(self.batch_size, self.board_size, self.board_size)
        # Losses
        policy_loss = -torch.sum(policies * log_probs) / self.batch_size
        value_loss = F.mse_loss(value_preds, values)
        total_loss = policy_loss + value_loss
        if round(policy_loss.item(), 4) == 0:
            print("Warn: policy_loss is 0")
        #print(policy_loss, value_loss)
        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

def tune_mcts():
    ## Use this to examine the UCB values and tune the values of c
    ## The first iteration should have UCB values spread around 0.4~0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("gomoku_gym/GridWorld-v0")
    board_size = env.unwrapped.board_size
    network = GomokuNet(board_size).to(device)
    network.load_state_dict(torch.load(f"gomoku_net_4x4_470-best.pth"))
    env.reset()
    env.step((2,1))
    env.step((2,3))
    env.step((0,1))
    env.step((0,3))
    env.unwrapped._render_frame()
    mcts = NetworkMCTS(env, env.unwrapped.player, network, board_size, 0)
    best_move = mcts.search_parallel(num_simulations=80) # 8 cores
    print(best_move)
    env.close()

if __name__ == "__main__":
    #tune_mcts()
    #raise ValueError()
    trainer = SelfPlayTrainer()
    print("Cuda:", torch.cuda.is_available())
    lowest_loss = 5
    for iteration in range(500):
        print(f"Iteration {iteration + trainer.iter_cont + 1}")
        #while trainer.p1done < 5 or trainer.p2done < 5:
            #print(trainer.p1done, trainer.p2done)
            #trainer.self_play(num_games=10)
        trainer.self_play(10)

        ave_loss = 0
        for _ in range(10):
            loss = trainer.train()
            ave_loss += loss
            print(f"Training loss: {loss:.4f}")
        ave_loss /= 10
        trainer.p1done = False
        trainer.p2done = False
        # Save model periodically
        if ave_loss < lowest_loss:
            torch.save(trainer.network.state_dict(), f"gomoku_net_7x7_{iteration+1+trainer.iter_cont}.pth")
            lowest_loss = ave_loss
        elif (iteration + 1) % 10 == 0:
            torch.save(trainer.network.state_dict(), f"gomoku_net_7x7_{iteration+1+trainer.iter_cont}.pth")