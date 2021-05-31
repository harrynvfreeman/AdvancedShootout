from game.agent.agent import Agent
import game.move
import numpy as np

class SmartAgent(Agent):
    def __init__(self, name="SmartAgent"):
        super().__init__(name)
        self.Q = np.load('/Users/HarryFreeman/Documents/Projects/AdvancedShootout/gym-advancedshotout/Q.npy')
        self.max_bullets = np.load('/Users/HarryFreeman/Documents/Projects/AdvancedShootout/gym-advancedshotout/max_bullets.npy')
    
    def get_next_action(self, opponent):
        s = np.min([self.max_bullets - 1, self.num_bullets]) * self.max_bullets + np.min([self.max_bullets - 1, opponent.num_bullets])
        action = game.move.move_dict[np.argmax(self.Q[s, :])]
        return action
    