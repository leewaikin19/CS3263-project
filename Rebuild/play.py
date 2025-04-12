import torch
from search import MCTS
from gomoku_env import Gomoku
from networks import ResNet
import numpy as np

tictactoe = Gomoku()
player = 1

args = {
    'C': 2,
    'num_searches': 800,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(tictactoe, 10, 128, device)
checkpoint = torch.load('model_12_300_Gomoku_5x5.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

mcts = MCTS(tictactoe, args, model)

state, x1, x2 = tictactoe.get_initial_state()


while True:
    print(state)
    # print(x1)
    # print(x2)

    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        #print("valid_moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        print("Enter (x, y): ")
        coord_input = input()
        x_str, y_str = coord_input.split()
        x = int(x_str)
        y = int(y_str)

        action = y * tictactoe.column_count + x

        if valid_moves[action] == 0:
            print("action not valid")
            continue

    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state, x1, x2)
        action = np.argmax(mcts_probs)

    state = tictactoe.get_next_state(state, action, player)
    next_p1 = x2
    next_p2 = tictactoe.step(action, x1)

    # print(next_p1)
    # print(next_p2)
    # print("xxxxx")
    value, is_terminal = tictactoe.get_value_and_terminated(state, next_p2)

    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break

    player = tictactoe.get_opponent(player)
    x1, x2 = next_p1, next_p2
