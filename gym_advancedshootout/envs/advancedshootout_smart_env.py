from game.env.advancedshootout_env import AdvancedShootoutEnv
from game.agent.smart_agent import SmartAgent
import numpy as np

class AdvancedShootoutSmartEnv(AdvancedShootoutEnv):
    def __init__(self):
        current_version = np.load('./train/version.npy')
        current_P_path = './train/' + str(current_version) + '/P.npy'
        current_max_bullets_path = './train/' + str(current_version) + '/max_bullets.npy'
        self.hidden_agent = SmartAgent(current_P_path, current_max_bullets_path)
        super().__init__()
    
    def set_deterministic(self, deterministic):
        self.hidden_agent.set_deterministic(deterministic)