import gymnasium as gym
import gomoku_gym 
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from networks import GomokuNet, board_to_tensor
from mcts_network import NetworkMCTS
import gc

class SelfPlayTrainer:
    def __init__(self):
        self.network = GomokuNet()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.env = gym.make("gomoku_gym/GridWorld-v0")  
        self.board_size = 15

        #self.network.load_state_dict(torch.load("gomoku_net_5.pth"))
    
    def self_play(self, num_games=10):
        env = self.env
        gc.collect()
        for i in range(num_games):
            print("Game", i+1)
            
            observation, info = env.reset()
            episode_over = False
            history = []
            
            mcts = NetworkMCTS(env, 1, self.network)
            
            while not episode_over:
                # Get MCTS policy
                root_node = mcts.root
                policy = np.zeros((15, 15))
                total_visits = sum(child.N for child in root_node.children.values())
                
                for move, child in root_node.children.items():
                    x, y = move
                    policy[y][x] = child.N / total_visits
                if total_visits > 0:
                    print(total_visits)
                    env.unwrapped._render_frame()
                # Store training data
                board_tensor = board_to_tensor(env, root_node.player)
                history.append((board_tensor, policy, root_node.player))
                
                # Make move
                # In self_play method:
                best_move = mcts.search(num_simulations=170)

                observation, reward, terminated, truncated, info = env.step(np.array(best_move, dtype=np.int32))
                mcts.move(best_move, env)
                episode_over = terminated or truncated
            env.unwrapped._render_frame()
            # Determine final reward
            if env.unwrapped._win(env.unwrapped._p1):
                final_value = 1
            elif env.unwrapped._win(env.unwrapped._p2):
                final_value = -1
            else:
                final_value = 0
            
            # Add to memory with appropriate value targets
            for (board_tensor, policy, player) in history:
                value_target = final_value if player == 1 else -final_value
                self.memory.append((board_tensor, policy, value_target))
            
            env.close()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        boards, policies, values = zip(*batch)
        
        boards = torch.cat(boards)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        # Forward pass
        self.optimizer.zero_grad()
        log_probs, value_preds = self.network(boards)
        # Losses
        policy_loss = -torch.sum(policies * log_probs.reshape(self.batch_size, self.board_size, self.board_size)) / self.batch_size
        value_loss = F.mse_loss(value_preds, values)
        total_loss = policy_loss + value_loss
        if np.close(policy_loss, 0):
            print("Warn: policy_loss is 0")
        #print(policy_loss, value_loss)
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

if __name__ == "__main__":
    trainer = SelfPlayTrainer()
    
    for iteration in range(100):
        print(f"Iteration {iteration + 1}")
        trainer.self_play(num_games=2)
        for _ in range(20):
            loss = trainer.train()
            print(f"Training loss: {loss:.4f}")
        
        # Save model periodically
        if (iteration + 1) % 5 == 0:
            torch.save(trainer.network.state_dict(), f"gomoku_net_{iteration+1}.pth")