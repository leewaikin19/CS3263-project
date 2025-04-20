import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class Gomoku:
    def __init__(self):
        self.board_size = 5
        self.row_count = self.board_size
        self.column_count = self.board_size
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return f"Gomoku_{self.board_size}x{self.board_size}"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count)), np.array([0] * self.board_size, dtype=np.uint16), np.array([0] * self.board_size, dtype=np.uint16)

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)


    def step(self, action, arr):
        # To prevent array copying, we change action[0] -> 14-action[0]
        row = action // self.column_count
        column = action % self.column_count
        action = (self.board_size-1-column, row)
        arr[action[1]] = arr[action[1]] | (1 << action[0])
        return arr
    
    """
    # ## 3-in-a-row to win
    def check_win(self, arr):
      hor = arr & (arr << 1) & (arr << 2)
      if np.count_nonzero(np.unpackbits(hor.view(np.uint8))) > 0:
          return True

      vert = arr[2:] & arr[1:-1] & arr[:-2]
      if np.count_nonzero(np.unpackbits(vert.view(np.uint8))) > 0:
          return True

      pos_diag = arr[2:] & (arr[1:-1] << 1) & (arr[:-2] << 2)
      if np.count_nonzero(np.unpackbits(pos_diag.view(np.uint8))) > 0:
          return True

      neg_diag = arr[:-2] & (arr[1:-1] << 1) & (arr[2:] << 2)
      if np.count_nonzero(np.unpackbits(neg_diag.view(np.uint8))) > 0:
          return True
      return False
    """ 
    
    def check_win(self, arr):
        hor = arr & (arr << 1) & (arr << 2) & (arr << 3)
        if np.count_nonzero(np.unpackbits(hor.view(np.uint8))) > 0:
            return True

        vert = arr[3:] & arr[2:-1] & arr[1:-2] & arr[:-3]
        if np.count_nonzero(np.unpackbits(vert.view(np.uint8))) > 0:
            return True

        pos_diag = arr[3:] & (arr[2:-1] << 1) & (arr[1:-2] << 2) & (arr[:-3] << 3)
        if np.count_nonzero(np.unpackbits(pos_diag.view(np.uint8))) > 0:
            return True

        neg_diag = arr[:-3] & (arr[1:-2] << 1) & (arr[2:-1] << 2) & (arr[3:] << 3)
        if np.count_nonzero(np.unpackbits(neg_diag.view(np.uint8))) > 0:
            return True
        return False
    
    
    """
    ### 5-in-a-row to win
    def check_win(self, arr):
        hor = arr & (arr << 1) & (arr << 2) & (arr << 3) & (arr << 4)
        if np.count_nonzero(np.unpackbits(hor.view(np.uint8))) > 0:
            return True

        vert = arr[4:] & arr[3:-1] & arr[2:-2] & arr[1:-3] & arr[:-4]
        if np.count_nonzero(np.unpackbits(vert.view(np.uint8))) > 0:
            return True

        pos_diag = arr[4:] & (arr[3:-1] << 1) & (arr[2:-2] << 2) & (arr[1:-3] << 3) & (arr[:-4] << 4)
        if np.count_nonzero(np.unpackbits(pos_diag.view(np.uint8))) > 0:
            return True

        neg_diag = arr[:-4] & (arr[1:-3] << 1) & (arr[2:-2] << 2) & (arr[3:-1] << 3) & (arr[4:] << 4)
        if np.count_nonzero(np.unpackbits(neg_diag.view(np.uint8))) > 0:
            return True
        return False
    """


    def get_value_and_terminated(self, state, arr):
        if self.check_win(arr):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
          encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
