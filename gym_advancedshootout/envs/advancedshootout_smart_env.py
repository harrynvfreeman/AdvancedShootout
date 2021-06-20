from game.env.advancedshootout_env import AdvancedShootoutEnv
from game.agent.smart_agent import SmartAgent
import numpy as np

class AdvancedShootoutSmartEnv(AdvancedShootoutEnv):
    hidden_agent_best = True
    hidden_agent_version = np.load('./train/version.npy')
    hidden_agent_deterministic = False
    def __init__(self):
        if AdvancedShootoutSmartEnv.hidden_agent_best:
            current_P_path = './train/best/P.npy'
        else:
            current_P_path = './train/' + str(AdvancedShootoutSmartEnv.hidden_agent_version) + '/P.npy'
        self.hidden_agent = SmartAgent(current_P_path, deterministic=AdvancedShootoutSmartEnv.hidden_agent_deterministic)
        super().__init__()
    
    @staticmethod
    def set_deterministic(deterministic):
        AdvancedShootoutSmartEnv.hidden_agent_deterministic = deterministic
    
    @staticmethod
    def set_version(version):
        if version is not None:
            AdvancedShootoutSmartEnv.hidden_agent_version = version
            
    @staticmethod
    def set_best(best):
        AdvancedShootoutSmartEnv.hidden_agent_best = best