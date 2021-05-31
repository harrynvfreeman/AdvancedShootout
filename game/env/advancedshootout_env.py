import gym
from gym import error, spaces, utils
from gym.utils import seeding
from game.agent.agent import Agent
from game.move import Move
import copy

class AdvancedShootoutEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        #self.hidden_agent
        self.external_agent_tracker = Agent()
        self.done = None
        self.reset()
        
    def step(self, action):
        if self.done:
            raise Exception("Cannot make move when done. Please reset.")
        
        #get next action before external agent updates itself
        hidden_agent_action = self.hidden_agent.get_next_action(self.external_agent_tracker)
        
        
        if action == Move.SHOOT and hidden_agent_action == Move.RELOAD:
            self.done = True
            reward = 1
        elif action == Move.RELOAD and hidden_agent_action == Move.SHOOT:
            self.done = True
            reward = -1
        else:
            reward = 0
        
        #now can update hidden and external agent
        self.hidden_agent.make_action(hidden_agent_action)
        self.external_agent_tracker.make_action(action)
        
        return self.get_observation(), reward, self.done, None
        
    def reset(self):
        self.hidden_agent.reset()
        self.external_agent_tracker.reset()
        self.done = False
        
    def render(self, mode='human'):
        pass
        
    def close(self):
        pass
    
    def get_observation(self):
        return copy.deepcopy(self.hidden_agent)
    