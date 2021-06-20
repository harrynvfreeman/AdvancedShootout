from game.agent.agent import Agent
import game.move
import numpy as np

class CheekyAgent(Agent):
    def __init__(self, name="CheekyAgent"):
        super().__init__(name)
    
    def get_next_action(self, opponent):
        if self.num_bullets == game.move.move_bullet_cost[game.move.Move.ROCKET.value]:
            return game.move.Move.ROCKET
        
        if self.num_bullets < 2:
            return game.move.Move.RELOAD
        
        if opponent.num_bullets > 0:
            return game.move.Move.SHIELD
        
        return game.move.Move.RELOAD
    
    def get_move_probs(self, opponent, max_bullets=None):
        pass