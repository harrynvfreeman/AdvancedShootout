from game.env.advancedshootout_env import AdvancedShootoutEnv
from game.agent.random_agent import RandomAgent

class AdvancedShootoutRandomEnv(AdvancedShootoutEnv):
    def __init__(self):
        self.hidden_agent = RandomAgent()
        super().__init__()
    