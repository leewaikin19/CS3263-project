import numpy as np
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
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(10)])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

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

    def board_to_tensor(self, p1, p2, current_player):
        """Convert gymnasium env state to neural network input tensor"""
        # Create binary boards
        board_p1 = torch.tensor(np.unpackbits(p1[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:]).to(torch.get_default_dtype())
        board_p2 = torch.tensor(np.unpackbits(p2[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:]).to(torch.get_default_dtype())
        player_ind = torch.ones((self.board_size, self.board_size)) * (1 if current_player == 1 else -1)

        # Stack into 3-channel tensor
        return torch.stack([board_p1, board_p2, player_ind], dim=0).unsqueeze(0).float()

    def board_to_tensor_symmetries(self, p1, p2, current_player, policy):
        boards = []
        # Create binary boards
        board_p1 = torch.tensor(np.unpackbits(p1[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:]).to(torch.get_default_dtype())
        board_p2 = torch.tensor(np.unpackbits(p2[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:]).to(torch.get_default_dtype())
        player_ind = torch.ones((self.board_size, self.board_size)) * (1 if current_player == 1 else -1)

        # rotate 0deg
        boards.append((torch.stack([board_p1, board_p2, player_ind], dim=0).unsqueeze(0).float(), policy))
        # rotate 90deg
        boards.append((torch.stack([torch.rot90(board_p1), torch.rot90(board_p2), player_ind], dim=0).unsqueeze(0).float(), np.rot90(policy)))
        # rotate 180deg
        boards.append((torch.stack([torch.rot90(board_p1, 2), torch.rot90(board_p2, 2), player_ind], dim=0).unsqueeze(0).float(), np.rot90(policy, 2)))
        # rotate 270deg
        boards.append((torch.stack([torch.rot90(board_p1, 3), torch.rot90(board_p2, 3), player_ind], dim=0).unsqueeze(0).float(), np.rot90(policy, 3)))
        # flip horizontally
        boards.append((torch.stack([torch.fliplr(board_p1), torch.fliplr(board_p2), player_ind], dim=0).unsqueeze(0).float(), np.fliplr(policy)))
        # flip vertically
        boards.append((torch.stack([torch.flipud(board_p1), torch.flipud(board_p2), player_ind], dim=0).unsqueeze(0).float(), np.flipud(policy)))
        # flip diagonally
        boards.append((torch.stack([torch.transpose(board_p1, 0, 1), torch.transpose(board_p2, 0, 1), player_ind], dim=0).unsqueeze(0).float(), np.transpose(policy)))
        # flip anti-diagonally
        boards.append((torch.stack([torch.transpose(torch.flipud(torch.fliplr(board_p1)), 0, 1), torch.transpose(torch.flipud(torch.fliplr(board_p2)), 0, 1), player_ind], dim=0).unsqueeze(0).float(), np.transpose(np.flipud(np.fliplr(policy)))))
        return boards