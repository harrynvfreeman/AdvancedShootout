import game.move
import numpy as np

input_dict = {"s": game.move.Move.SHIELD,
              "r": game.move.Move.RELOAD,
              "f": game.move.Move.SHOOT}

class HumanAgent:
    def __init__(self, name="HumanAgent"):
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
        if self.num_bullets == 0:
            input_string = "Enter move (s-shield, r-reload): "
        else:
            input_string = "Enter move (s-shield, r-reload, f-shoot): "
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
            
        if action == game.move.Move.RELOAD:
            self.num_bullets = self.num_bullets + 1
            self.valid_moves[game.move.Move.SHOOT.value] = 1
        elif action == game.move.Move.SHOOT:
            self.num_bullets = self.num_bullets - 1
            if self.num_bullets == 0:
                self.valid_moves[game.move.Move.SHOOT.value] = 0
        self.last_action = action
        return action
    