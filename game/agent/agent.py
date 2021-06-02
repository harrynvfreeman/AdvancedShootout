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
        self.update_valid_actions()
        self.last_action = None
        
    def get_valid_actions(self):
        return np.copy(self.valid_moves)
    
    def get_is_alive(self):
        return self.is_alive
    
    def make_action(self, action):
        if self.valid_moves[action.value] == 0:
            raise Exception(self.name + " made illegal action " + str(action) + " with " + str(self.num_bullets) + " bullets.")
        
        self.num_bullets = self.num_bullets + self.get_bullet_diff(action)
        self.update_valid_actions()
        self.last_action = action
        
    def update_valid_actions(self):
        self.valid_moves[game.move.Move.SHOOT.value] =  self.num_bullets >= 1
        self.valid_moves[game.move.Move.SHOTGUN.value] =  self.num_bullets >= 2
        self.valid_moves[game.move.Move.ROCKET.value] =  self.num_bullets >= 4
    
    def force_num_bullets(self, num_bullets):
        if num_bullets < 0:
            raise Exception("Cannot set num bullets to " + str(num_bullets) + " for " + self.name)
        
        self.num_bullets = num_bullets
        self.update_valid_actions()
        
    def get_bullet_diff(self, action):
        if action == game.move.Move.SHIELD:
            return 0
        if action == game.move.Move.RELOAD:
            return 1
        if action == game.move.Move.SHOOT:
            return -1
        if action == game.move.Move.SHOTGUN:
            return -2
        if action == game.move.Move.ROCKET:
            return -4
    
    def get_next_action(self, opponent=None):
        pass
    
    def get_move_probs(self, opponent, max_bullets=None):
        pass