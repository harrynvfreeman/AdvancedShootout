from game.agent.agent import Agent
import numpy as np
from mcts.mcts import Tree

class MctsAgent(Agent):
    def __init__(self, path, name="MctsAgent"):
        self.V = np.load(path + '/V.npy')
        self.P = np.load(path + '/P.npy')
        super().__init__(name)
        
        
    
    def get_next_action(self, opponent):
        action, _ = self.tree.play_instance_get_move()
        return action
    
    def post_move_update(self, action, op_action):
        self.tree.update_state(action, op_action, None)
        
    def reset(self):
        super().reset()
        self.tree = Tree(self.V, self.P)