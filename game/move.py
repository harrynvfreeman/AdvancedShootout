from enum import Enum

num_moves = 3

class Move(Enum):
    SHIELD = 0
    RELOAD = 1
    SHOOT = 2
    
move_dict = {Move.SHIELD.value: Move.SHIELD,
             Move.RELOAD.value: Move.RELOAD,
             Move.SHOOT.value: Move.SHOOT}