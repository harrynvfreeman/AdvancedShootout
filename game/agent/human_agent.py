from game.agent.agent import Agent
import game.move
import numpy as np

input_dict = {"s": game.move.Move.SHIELD,
              "r": game.move.Move.RELOAD,
              "f": game.move.Move.SHOOT,
              "g": game.move.Move.SHOTGUN,
              "v": game.move.Move.ROCKET,
              "b": game.move.Move.SONIC_BOOM}

class HumanAgent(Agent):
    def __init__(self, name="HumanAgent"):
        super().__init__(name)
        
    def get_next_action(self, opponent=None):
        if self.num_bullets < game.move.move_bullet_cost[game.move.Move.SHOOT.value]:
            input_string = "Enter move (s-shield, r-reload): "
        elif self.num_bullets < game.move.move_bullet_cost[game.move.Move.SHOTGUN.value]:
            input_string = "Enter move (s-shield, r-reload, f-shoot): "
        elif self.num_bullets < game.move.move_bullet_cost[game.move.Move.ROCKET.value]:
            input_string = "Enter move (s-shield, r-reload, f-shoot, g-shotgun): "
        elif self.num_bullets < game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]:
            input_string = "Enter move (s-shield, r-reload, f-shoot, g-shotgun, v-rocket): "
        else:
            input_string = "Enter move (s-shield, r-reload, f-shoot, g-shotgun, v-rocket, b-sonic_boom): "
            
        valid_input = False
        while not valid_input:
            input_action = input(input_string)
            if input_dict.get(input_action) == None:
                print("Invalid move entered, try again")
                continue
            action = input_dict.get(input_action)
            if self.valid_moves[action.value] == 0:
                print("Invalid move entered, try again")
                continue
            valid_input = True
        return action
    
    def get_move_probs(self, opponent, max_bullets=None):
        pass
    