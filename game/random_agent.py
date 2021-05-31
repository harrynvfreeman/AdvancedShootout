import game.move
import numpy as np

class RandomAgent:
    def __init__(self, name="RandomAgent"):
        self.name = name
        self.num_bullets = None
        self.valid_moves = None
        self.last_action = None
        self.reset()
        
    def reset(self):
        self.num_bullets = 0
        self.valid_moves = np.ones((game.move.num_moves))
        self.valid_moves[game.move.Move.SHOOT.value] = 0
        self.last_action = None
        
    def get_action(self, opponent=None):
        num_valid = np.sum(self.valid_moves)
        action = game.move.move_dict[np.random.choice(game.move.num_moves, 1, p=1/num_valid * self.valid_moves)[0]]
        if action == game.move.Move.RELOAD:
            self.num_bullets = self.num_bullets + 1
            self.valid_moves[game.move.Move.SHOOT.value] = 1
        elif action == game.move.Move.SHOOT:
            self.num_bullets = self.num_bullets - 1
            if self.num_bullets == 0:
                self.valid_moves[game.move.Move.SHOOT.value] = 0
        self.last_action = action
        return action
    