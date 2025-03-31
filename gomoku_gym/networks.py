import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class GomokuNet(nn.Module):
    def __init__(self, board_size=15):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        #Residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(0)])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # WARN DISABLE THIS IF LOADING SAVED WEIGHTS
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights of each layer with a normal distribution
        init.normal_(self.conv1.weight, mean=0, std=0.1) 
        init.normal_(self.conv2.weight, mean=0, std=0.1)  
        init.normal_(self.conv3.weight, mean=0, std=0.1)  

        init.normal_(self.policy_conv.weight, mean=0, std=0.1) 
        init.normal_(self.policy_fc.weight, mean=0, std=0.1)  
        init.normal_(self.value_conv.weight, mean=0, std=0.1)  
        init.normal_(self.value_fc1.weight, mean=0, std=0.1)  
        init.normal_(self.value_fc2.weight, mean=0, std=0.1)  
        

    def forward(self, x):
        # Input shape: (batch_size, 3, board_size, board_size)
        # Channel 0: Player 1 stones
        # Channel 1: Player 2 stones
        # Channel 2: Current player indicator
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        for res_block in self.residual_blocks:
            x = res_block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * self.board_size * self.board_size)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
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