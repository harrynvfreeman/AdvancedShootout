from game.agent.agent import Agent
import game.move
import numpy as np

class DumbAgent(Agent):
    def __init__(self, name="DumbAgent"):
        super().__init__(name)
    
    def get_next_action(self, opponent):
        if self.num_bullets == 4:
            return game.move.Move.ROCKET
        else:
            return game.move.Move.RELOAD
    
    def get_move_probs(self, opponent, max_bullets=None):
        pass