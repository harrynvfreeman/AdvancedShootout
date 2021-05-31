from game.env.advancedshootout_env import AdvancedShootoutEnv
from game.agent.smart_agent import SmartAgent

class AdvancedShootoutSmartEnv(AdvancedShootoutEnv):
    def __init__(self):
        self.hidden_agent = SmartAgent()
        super().__init__()
    