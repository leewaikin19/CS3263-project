import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import hashlib


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.board_size = 7
        self.player = 1
        self._p1 = np.array([0] * self.board_size, dtype=np.uint16)
        self._p2 = np.array([0] * self.board_size, dtype=np.uint16)
        self.history = []
        self.moves = 0
        self.action_space = spaces.MultiDiscrete([self.board_size, self.board_size])
        self.observation_space = spaces.Dict({"p1": spaces.Box(0, 2**self.board_size - 1, shape=(self.board_size,), dtype=np.uint16), "p2": spaces.Box(0, 2**self.board_size - 1, shape=(self.board_size,), dtype=np.uint16)})

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {"p1": self._p1, "p2": self._p2}

    def _get_info(self):
        return {
            "placeholder": "some info"
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set up a 15 x 15 board
        self.player = 1
        self._p1 = np.array([0] * self.board_size, dtype=np.uint16)
        self._p2 = np.array([0] * self.board_size, dtype=np.uint16)
        self.history = []
        self.moves = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # To prevent array copying, we change action[0] -> 14-action[0]
        action = (self.board_size-1-action[0], action[1])

        # Ensure the action is valid
        valid = (((self._p1[action[1]] >> action[0]) & 1) == 0) and ((self._p2[action[1]] >> action[0]) & 1) == 0

        if valid:
            # print(action, self.player)
            self.history.append((self.player, action))
            if self.player == 1:
                self._p1[action[1]] = self._p1[action[1]] | (1 << action[0])
            elif self.player == 2:
                self._p2[action[1]] = self._p2[action[1]] | (1 << action[0])
            self.player = self.player % 2 + 1
            self.moves += 1


        num_bits = 0
        for i in range(self.board_size):
            num_bits += self._p1[i].bit_count()
            num_bits += self._p2[i].bit_count()

        terminated = False
        p1_win = self._win(self._p1)
        if p1_win:
            terminated = True
            reward = 1 == self.player
        p2_win = self._win(self._p2)
        if p2_win:
            terminated = True
            reward = 2 == self.player
        if(num_bits == self.board_size*self.board_size):
            terminated = True
            reward = 0


        if terminated and self.render_mode == "human":
            self._render_frame()

        reward = 0

        observation = self._get_obs()
        info = {"valid": valid}

        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def sim_step(self, _p1, _p2, player, action):
        # To prevent array copying, we change action[0] -> 14-action[0]
        action = (self.board_size-1-action[0], action[1])

        valid = (((_p1[action[1]] >> action[0]) & 1) == 0) and ((_p2[action[1]] >> action[0]) & 1) == 0

        if valid:
            if player == 1:
                _p1[action[1]] = _p1[action[1]] | (1 << action[0])
                return _p1
            elif player == 2:
                _p2[action[1]] = _p2[action[1]] | (1 << action[0])
                return _p2
        raise ValueError()

    def undo(self):
        player, action = self.history.pop() # The action here is correct, no need to flip
        self.player = 3 - player
        if player == 1:
            self._p1[action[1]] = self._p1[action[1]] & ~(1 << action[0])
        else:
            self._p2[action[1]] = self._p2[action[1]] & ~(1 << action[0])

    def clone(self):
        new_env = GridWorldEnv()  # Create a new instance of the environment
        new_env.player = self.player
        new_env._p1 = copy.deepcopy(self._p1)
        new_env._p2 = copy.deepcopy(self._p2)
        return new_env

    ### 3-in-a-row to win
    #def _win(self, arr):
    #    hor = arr & (arr << 1) & (arr << 2)
    #    if np.count_nonzero(np.unpackbits(hor.view(np.uint8))) > 0:
    #        return True

    #    vert = arr[2:] & arr[1:-1] & arr[:-2]
    #    if np.count_nonzero(np.unpackbits(vert.view(np.uint8))) > 0:
    #        return True

    #    pos_diag = arr[2:] & (arr[1:-1] << 1) & (arr[:-2] << 2)
    #    if np.count_nonzero(np.unpackbits(pos_diag.view(np.uint8))) > 0:
    #        return True

    #   neg_diag = arr[:-2] & (arr[1:-1] << 1) & (arr[2:] << 2)
    #   if np.count_nonzero(np.unpackbits(neg_diag.view(np.uint8))) > 0:
    #        return True
    #    return False

    ### 4-in-a-row to win
    def _win(self, arr):
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

    ### 5-in-a-row to win
    # def _win(self, arr):
    #     hor = arr & (arr << 1) & (arr << 2) & (arr << 3) & (arr << 4)
    #     if np.count_nonzero(np.unpackbits(hor.view(np.uint8))) > 0:
    #         return True

    #     vert = arr[4:] & arr[3:-1] & arr[2:-2] & arr[1:-3] & arr[:-4]
    #     if np.count_nonzero(np.unpackbits(vert.view(np.uint8))) > 0:
    #         return True

    #     pos_diag = arr[4:] & (arr[3:-1] << 1) & (arr[2:-2] << 2) & (arr[1:-3] << 3) & (arr[:-4] << 4)
    #     if np.count_nonzero(np.unpackbits(pos_diag.view(np.uint8))) > 0:
    #         return True

    #     neg_diag = arr[:-4] & (arr[1:-3] << 1) & (arr[2:-2] << 2) & (arr[3:-1] << 3) & (arr[4:] << 4)
    #     if np.count_nonzero(np.unpackbits(neg_diag.view(np.uint8))) > 0:
    #         return True
    #     return False

    def _render_frame(self, _p1=None, _p2=None, player=None):
        if np.all(_p1 == None) or np.all(_p2 == None):
            _p1 = self._p1
            _p2 = self._p2
            player = self.player
        if self._win(_p1):
            print("   Player 1 won!" + " " * (self.board_size*2 - 12) + "Idx")
        elif self._win(_p2):
            print("   Player 2 won!" + " " * (self.board_size*2 - 12) + "Idx")
        elif self.moves == self.board_size * self.board_size:
            print("   Draw!" + " " * (self.board_size*2 - 4) + "Idx")
        else:
            print("   Player", str(player) + "'s turn", "                Idx")
        combined = np.unpackbits(_p1[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:] + np.unpackbits(_p2[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:] * 2
        for row in range(self.board_size):
            #combined = np.array([int(char) for char in format(_p1[14-row], "016b")])[1:] + np.array([int(char) for char in format(_p2[14-row], "016b")])[1:]*2
            print("  ", combined[row], row)
        print("Idx 0 1 2 3 4 5 6 7 8 9 1 1 1 1 1")
        print("                        0 1 2 3 4")

    def _get_valid_moves(self, _p1=None, _p2=None):
        if np.all(_p1 == None) or np.all(_p2 == None):
            _p1 = self._p1
            _p2 = self._p2
        combined = np.unpackbits(_p1[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:] + np.unpackbits(_p2[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:]
        row, col = np.where(combined == 0)
        valid_moves = np.column_stack((col, row))
        #for row in range(15):
        #    for col in range(15):
        #        if (((_p1[row] >> col) & 1) == 0) and (((_p2[row] >> col) & 1) == 0):
        #            valid_moves.append((14 - col, row))
        return valid_moves

    def _get_valid_mask(self, _p1=None, _p2=None):
        if np.all(_p1 == None) or np.all(_p2 == None):
            _p1 = self._p1
            _p2 = self._p2
        combined = np.unpackbits(_p1[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:] + np.unpackbits(_p2[:, np.newaxis].byteswap().view(np.uint8), axis=1)[:, 16-self.board_size:]
        combined = (combined - 1)* -1
        return combined

    def hash(self, _p1, _p2, player):
        if np.all(_p1 == None) or np.all(_p2 == None):
            _p1 = self._p1
            _p2 = self._p2
            player = self.player

        hash_obj = hashlib.sha512()
        hash_obj.update(_p1.tobytes())
        hash_obj.update(_p2.tobytes())
        hash_obj.update(player.to_bytes(8, byteorder='little'))

        return hash_obj.hexdigest()