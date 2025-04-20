from gomoku_gym.envs.gomoku_world import GridWorldEnv

# added the following lines

from gymnasium.envs.registration import register

register(
    id="gomoku_gym/GridWorld-v0",  # Keep this ID to match what works
    entry_point="gomoku_gym.envs.gomoku_world:GridWorldEnv",
    kwargs={"size": 15}
)