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
        for i in range(self.valid_moves.shape[0]):
            self.valid_moves[i] = self.num_bullets >= game.move.move_bullet_cost[i]
    
    def force_num_bullets(self, num_bullets):
        if num_bullets < 0:
            raise Exception("Cannot set num bullets to " + str(num_bullets) + " for " + self.name)
        
        self.num_bullets = num_bullets
        self.update_valid_actions()
        
    def get_bullet_diff(self, action):
        return game.move.move_bullet_gain[action.value] - game.move.move_bullet_cost[action.value]
    
    def get_next_action(self, opponent=None):
        pass
    
    def get_move_probs(self, opponent, max_bullets=None):
        pass
    
    def post_move_update(self, action, op_action):
        pass