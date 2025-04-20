from gomoku_env import Gomoku
import torch
from networks import ResNet
from learner import ParallelLearner

game = Gomoku()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 12, 128, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001) #added weight decay

args = {
    'C': 2,
    'num_searches': 500, #this is extreme, currently training a 7x7, set for 5 days
    'num_iterations': 500000,
    'num_selfPlay_iterations': 1200,
    'num_parallel_games' : 400,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.15,
    'dirichlet_alpha': 0.3
}

parallellearner = ParallelLearner(model, game, optimizer, args)
parallellearner.learn()
