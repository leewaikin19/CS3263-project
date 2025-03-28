import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNet(nn.Module):
    def __init__(self, board_size=15):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Input shape: (batch_size, 3, board_size, board_size)
        # Channel 0: Player 1 stones
        # Channel 1: Player 2 stones
        # Channel 2: Current player indicator
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(-1, 2 * self.board_size * self.board_size)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(-1, self.board_size * self.board_size)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # Output between -1 and 1
        
        return p, v

def board_to_tensor(env, current_player):
    """Convert gymnasium env state to neural network input tensor"""
    p1 = env.unwrapped._p1
    p2 = env.unwrapped._p2
    
    # Create binary boards
    board_p1 = torch.zeros((15, 15))
    board_p2 = torch.zeros((15, 15))
    player_ind = torch.ones((15, 15)) * (1 if current_player == 1 else -1)
    
    for y in range(15):
        for x in range(15):
            if (p1[y] >> x) & 1:
                board_p1[y, x] = 1
            if (p2[y] >> x) & 1:
                board_p2[y, x] = 1
    
    # Stack into 3-channel tensor
    return torch.stack([board_p1, board_p2, player_ind], dim=0).unsqueeze(0).float()