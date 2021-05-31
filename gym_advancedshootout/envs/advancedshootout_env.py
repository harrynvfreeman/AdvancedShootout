import gym
from gym import error, spaces, utils
from gym.utils import seeding
from game.move import Move

class AdvancedShootoutEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.character_a = Character("A")
        self.character_b = Character("B")
        
    def step(self, action):
        a_move = action.a_move
        b_move = action.b_move
        
        self.character_a.step(a_move, b_move)
        self.character_b.step(b_move, a_move)
        
        #observation, reward, done, info
        done = not(self.character_a.is_alive and self.character_b.is_alive)
        return self.character_a, self.character_b, done, None
        
    def reset(self):
        self.character_a.reset()
        self.character_b.reset()
        
    def render(self, mode='human'):
        pass
        
    def close(self):
        pass