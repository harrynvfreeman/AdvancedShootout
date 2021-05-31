import game.move
import numpy as np

input_dict = {"s": game.move.Move.SHIELD,
              "r": game.move.Move.RELOAD,
              "f": game.move.Move.SHOOT}

class SmartAgent:
    def __init__(self, name="HumanAgent"):
        self.name = name
        self.num_bullets = None
        self.valid_moves = None
        self.last_action = None
        self.Q = np.load('/Users/HarryFreeman/Documents/Projects/AdvancedShootout/gym-advancedshotout/Q.npy')
        self.max_bullets = 20
        self.reset()
        
    def reset(self):
        self.num_bullets = 0
        self.valid_moves = np.ones((game.move.num_moves))
        self.valid_moves[game.move.Move.SHOOT.value] = 0
        self.last_action = None
        
    def get_action(self, opponent):
        s = np.min([self.max_bullets - 1, self.num_bullets]) * self.max_bullets + np.min([self.max_bullets - 1, opponent.num_bullets])
        
        action = game.move.move_dict[np.argmax(self.Q[s, :])]
        
        if action == game.move.Move.RELOAD:
            self.num_bullets = self.num_bullets + 1
            self.valid_moves[game.move.Move.SHOOT.value] = 1
        elif action == game.move.Move.SHOOT:
            self.num_bullets = self.num_bullets - 1
            if self.num_bullets == 0:
                self.valid_moves[game.move.Move.SHOOT.value] = 0
        self.last_action = action
        return action
    