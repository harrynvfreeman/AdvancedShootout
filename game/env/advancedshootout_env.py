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
        
        reward = get_reward(action, hidden_agent_action)
        if reward != 0:
            self.done = True
        else:
            self.done = False
        
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
    
def get_reward(action_a, action_b):
    if action_a not in [Move.SHIELD, Move.RELOAD, Move.SHOOT, Move.SHOTGUN, Move.ROCKET, Move.SONIC_BOOM]:
        raise Exception("Illegal action_a detected: " + str(action_a))
    if action_b not in [Move.SHIELD, Move.RELOAD, Move.SHOOT, Move.SHOTGUN, Move.ROCKET, Move.SONIC_BOOM]:
        raise Exception("Illegal action_b detected: " + str(action_b))
    
    #sonic boom case
    if action_a == Move.SONIC_BOOM and action_b != Move.SONIC_BOOM:
        return 1
    if action_a != Move.SONIC_BOOM and action_b == Move.SONIC_BOOM:
        return -1
    if action_a == Move.SONIC_BOOM and action_b == Move.SONIC_BOOM:
        return 0
    
    #if a shields and b rockets, a loses
    #otherwise, game continues
    if action_a == Move.SHIELD:
        if action_b == Move.ROCKET:
            return -1
        else:
            return 0
    #if a reloads and b shoots, shotguns, or rockets, a loses
    #otherwise if a reloads game continues 
    elif action_a == Move.RELOAD:
        if action_b in [Move.SHOOT, Move.SHOTGUN, Move.ROCKET]:
            return -1
        else:
            return 0
    #if a shoots and b shoots or shields, game continues
    #if a shoots and b reloads, a wins
    #if a shoots and b shotguns or rockets, a loses
    elif action_a == Move.SHOOT:
        if action_b in [Move.SHOOT, Move.SHIELD]:
            return 0
        elif action_b == Move.RELOAD:
            return 1
        else:
            return -1
    #if a shotguns and b shields, shotguns, or rockets, game continues
    #if a shotguns and b reloads, or shoots, a wins
    elif action_a == Move.SHOTGUN:
        if action_b in [Move.SHIELD, Move.SHOTGUN, Move.ROCKET]:
            return 0
        else:
            return 1
    #if a rockets and b shotguns or rockets, game continues
    #otherwise a wins
    else:
        if action_b in [Move.SHOTGUN, Move.ROCKET]:
            return 0
        else:
            return 1
        