import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, size=5):
        self.player = 1
        self._p1 = np.array([0] * 15, dtype=np.uint16)
        self._p2 = np.array([0] * 15, dtype=np.uint16)
        self.history = []
        self.action_space = spaces.MultiDiscrete([15, 15])
        self.observation_space = spaces.Dict({"p1": spaces.Box(0, 2**15 - 1, shape=(15,), dtype=np.uint16), "p2": spaces.Box(0, 2**15 - 1, shape=(15,), dtype=np.uint16)})

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
        self.player = self.__old_player = 1
        self._p1 = self.__old_p1 = np.array([0] * 15, dtype=np.uint16)
        self._p2 = self.__old_p2 = np.array([0] * 15, dtype=np.uint16)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # copy to __old first
        self.history.append((self.player, action))

        # Ensure the action is valid
        valid = (((self._p1[action[1]] >> action[0]) & 1) == 0) and ((self._p2[action[1]] >> action[0]) & 1) == 0

        if valid:
            # print(action, self.player)
            if self.player == 1:
                self._p1[action[1]] = self._p1[action[1]] | (1 << action[0])
            elif self.player == 2:
                self._p2[action[1]] = self._p2[action[1]] | (1 << action[0])
            self.player = self.player % 2 + 1
        
        num_bits = 0
        for i in range(15):
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
        if(num_bits == 225):
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
    
    def undo(self):
        player, action = self.history.pop()
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

    def _win(self, arr):
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
        
    def _render_frame(self):
        print("   Player", str(self.player) + "'s turn", "                Idx")
        for row in range(15):
            combined = np.array([int(char) for char in format(self._p1[14-row], "016b")])[1:] + np.array([int(char) for char in format(self._p2[14-row], "016b")])[1:]*2
            print("  ", combined, 14-row)
        print("Idx 1 1 1 1 1 9 8 7 6 5 4 3 2 1 0")
        print("    4 3 2 1 0")

    def _get_valid_moves(env):
        valid_moves = []
        for row in range(15):
            for col in range(15):
                if (((env._p1[row] >> col) & 1) == 0) and (((env._p2[row] >> col) & 1) == 0):
                    valid_moves.append((col, row))
        return valid_moves
