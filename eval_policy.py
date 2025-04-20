import torch
from gomoku_env import Gomoku
from networks import ResNet
import matplotlib.pyplot as plt

game = Gomoku()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state, xx1, xx2 = game.get_initial_state()
state = game.get_next_state(state, 2, -1)
state = game.get_next_state(state, 4, -1)
state = game.get_next_state(state, 6, 1)
state = game.get_next_state(state, 8, 1)


encoded_state = game.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(game, 10, 128, device)
model.load_state_dict(torch.load('model_4_Gomoku_3x3.pt', map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(policy)
print(value)

print(state)
print(tensor_state)

plt.bar(range(game.action_size), policy)
plt.show()

#this must confidently show (i.e prob ~0.9) that it must place the next dot at action 7 (this is a flat action)
