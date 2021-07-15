from game.env.advancedshootout_env import AdvancedShootoutEnv
from game.agent.mcts_agent import MctsAgent
import numpy as np

class AdvancedShootoutMctsEnv(AdvancedShootoutEnv):
    hidden_agent_best = True
    hidden_agent_version = np.load('./train/version.npy')
    def __init__(self):
        if AdvancedShootoutMctsEnv.hidden_agent_best:
            best_version = np.load('./train/best/best_version.npy')
            path = './train/best/' + str(best_version)
        else:
            path = './train/' + str(AdvancedShootoutMctsEnv.hidden_agent_version)
        self.hidden_agent = MctsAgent(path)
        super().__init__()
        
        
    @staticmethod
    def set_version(version):
        if version is not None:
            AdvancedShootoutMctsEnv.hidden_agent_version = version
            
    @staticmethod
    def set_best(best):
        AdvancedShootoutMctsEnv.hidden_agent_best = best