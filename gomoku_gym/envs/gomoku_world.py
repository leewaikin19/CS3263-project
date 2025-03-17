import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Action:
    pass

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, size=5):
        self.player = 1
        self._board = np.array([[0]*15] * 15, dtype=np.int64)
        self.action_space = spaces.MultiDiscrete([15, 15])
        self.observation_space = spaces.Box(0, 2, shape=(15,15), dtype=np.int64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self._board

    def _get_info(self):
        return {
            "placeholder": "some info"
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set up a 15 x 15 board
        self._board = np.array([[0]*15] * 15)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, params):
        # Ensure the action is valid
        action = params[0:2]
        valid = self._board[action[0]][action[1]] == 0

        if valid:
            print(action, self.player)
            self._board[action[0]][action[1]] = self.player
            self.player = self.player % 2 + 1
        
        # Add terminating condition later, placeholder for full board (draw)
        terminated = not np.any(self._board == 0)
        if terminated:
            print(self._board)

        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = {"valid": valid}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        print(self._board)