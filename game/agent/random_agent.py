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
    
    def get_move_probs(self, opponent, max_bullets=None):
        valid_actions = self.get_valid_actions()
        if (max_bullets is not None) and (self.num_bullets >= max_bullets - 1):
            valid_actions[game.move.Move.RELOAD.value] = 0
        return (1.0 / np.sum(valid_actions)) * valid_actions
    