from game.agent.agent import Agent
import game.move
import numpy as np

class SmartAgent(Agent):
    def __init__(self, P_path, deterministic=True, name="SmartAgent"):
        super().__init__(name)
        self.P = np.load(P_path)
        self.max_bullets = game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]
        self.deterministic = deterministic
    
    def get_next_action(self, opponent):
        s = np.min([self.max_bullets - 1, self.num_bullets]) * self.max_bullets + np.min([self.max_bullets - 1, opponent.num_bullets])
        if self.deterministic:
            action = game.move.move_dict[np.argmax(self.P[s, :])]
        else:
            action = game.move.move_dict[np.random.choice(game.move.num_moves, 1, p=1/self.P[s,:].sum()*self.P[s,:])[0]]
        return action
    
    def get_move_probs(self, opponent, max_bullets=None):
        action = self.get_next_action(opponent)
        if self.get_valid_actions()[action.value] == 0:
            raise Exception("smart agent somehow chose illegal action " + str(action))
        if self.deterministic:
            probs = np.zeros((game.move.num_moves))
            probs[action.value] = 1.0
        else:
            s = np.min([self.max_bullets - 1, self.num_bullets]) * self.max_bullets + np.min([self.max_bullets - 1, opponent.num_bullets])
            probs = self.P[s,:].copy()
        return probs