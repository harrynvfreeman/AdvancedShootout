from game.agent.agent import Agent
import game.move
import numpy as np

class SmartAgent(Agent):
    def __init__(self, Q_path, max_bullets_path, name="SmartAgent"):
        super().__init__(name)
        self.Q = np.load(Q_path)
        self.max_bullets = np.load(max_bullets_path)
    
    def get_next_action(self, opponent):
        s = np.min([self.max_bullets - 1, self.num_bullets]) * self.max_bullets + np.min([self.max_bullets - 1, opponent.num_bullets])
        action = game.move.move_dict[np.argmax(self.Q[s, :])]
        return action
    
    def get_move_probs(self, opponent, max_bullets=None):
        action = self.get_next_action(opponent)
        if self.get_valid_actions()[action.value] == 0:
            raise Exception("smart agent somehow chose illegal action " + str(action))
        probs = np.zeros((game.move.num_moves))
        probs[action.value] = 1.0
        return probs