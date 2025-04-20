import torch
from search import MCTS
from gomoku_env import Gomoku
from networks import ResNet
import numpy as np

game = Gomoku()
player = 1
#player = -1

args = {
    'C': 2,
    'num_searches': 800,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.15,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 10, 128, device)
#checkpoint = torch.load('weights/new_model_30_500_Gomoku_5x5.pt', map_location=device)
checkpoint = torch.load('weights/model_43_150_Gomoku_5x5.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

mcts = MCTS(game, args, model)

state, x1, x2 = game.get_initial_state()


while True:
    print(state)
    # print(x1)
    # print(x2)

    if player == 1:
        valid_moves = game.get_valid_moves(state)
        #print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
        print("Enter (x, y): ")
        coord_input = input()
        x_str, y_str = coord_input.split()
        x = int(x_str)
        y = int(y_str)

        action = y * game.column_count + x

        if valid_moves[action] == 0:
            print("action not valid")
            continue

    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state, x1, x2)
        action = np.argmax(mcts_probs)

    state = game.get_next_state(state, action, player)
    next_p1 = x2
    next_p2 = game.step(action, x1)

    # print(next_p1)
    # print(next_p2)
    # print("xxxxx")
    value, is_terminal = game.get_value_and_terminated(state, next_p2)

    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = game.get_opponent(player)
    x1, x2 = next_p1, next_p2
