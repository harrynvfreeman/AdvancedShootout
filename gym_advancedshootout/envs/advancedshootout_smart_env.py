import gym
from gym import error, spaces, utils
from gym.utils import seeding
from game.smart_agent import SmartAgent
from game.move import Move
import copy

class AdvancedShootoutSmartEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.hidden_agent = SmartAgent()
        self.done = None
        self.reset()
        
    def step(self, action, opponent):
        if self.done:
            raise Exception("Cannot make move when done. Please reset.")
        hidden_agent_move = self.hidden_agent.get_action(opponent)
        outside_agent_move = action
        
        if outside_agent_move == Move.SHOOT and hidden_agent_move == Move.RELOAD:
            self.done = True
            reward = 1
        elif outside_agent_move == Move.RELOAD and hidden_agent_move == Move.SHOOT:
            self.done = True
            reward = -1
        else:
            reward = 0
        
        
        return self.get_observation(), reward, self.done, None
        
    def reset(self):
        self.hidden_agent.reset()
        self.done = False
        
    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    
    def get_observation(self):
        return copy.deepcopy(self.hidden_agent)
    