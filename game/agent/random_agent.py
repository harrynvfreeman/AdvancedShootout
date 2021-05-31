from game.agent.agent import Agent
import game.move
import numpy as np

class RandomAgent(Agent):
    def __init__(self, name="RandomAgent"):
        super().__init__(name)
    
    def get_next_action(self, opponent=None):
        num_valid = np.sum(self.valid_moves)
        action = game.move.move_dict[np.random.choice(game.move.num_moves, 1, p=1/num_valid * self.valid_moves)[0]]
        return action
    