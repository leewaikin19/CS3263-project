from gymnasium.envs.registration import register

register(
    id="gomoku_gym/GridWorld-v0",
    entry_point="gomoku_gym.envs:GridWorldEnv",
)
