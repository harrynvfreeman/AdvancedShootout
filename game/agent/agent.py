import game.move
import numpy as np

class Agent:
    def __init__(self, name="Agent"):
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
        
    def get_valid_actions(self):
        return np.copy(self.valid_moves)
    
    def get_is_alive(self):
        return self.is_alive
    
    def make_action(self, action):
        if action == game.move.Move.SHIELD:
            pass
        elif action == game.move.Move.RELOAD:
            self.num_bullets = self.num_bullets + 1
            self.valid_moves[game.move.Move.SHOOT.value] = 1
        elif action == game.move.Move.SHOOT:
            self.num_bullets = self.num_bullets - 1
            if self.num_bullets == 0:
                self.valid_moves[game.move.Move.SHOOT.value] = 0
        self.last_action = action
        
    def force_num_bullets(self, num_bullets):
        if num_bullets < 0:
            raise Exception("Cannot set num bullets to " + str(num_bullets) + " for " + self.name)
        
        self.num_bullets = num_bullets
        if self.num_bullets > 0:
            self.valid_moves[game.move.Move.SHOOT.value] = 1
        else:
            self.valid_moves[game.move.Move.SHOOT.value] = 0
        
    def get_next_action(self, opponent=None):
        pass