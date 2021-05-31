from enum import Enum

num_moves = 5

class Move(Enum):
    SHIELD = 0
    RELOAD = 1
    SHOOT = 2
    SHOTGUN = 3
    ROCKET = 4
    
move_dict = {Move.SHIELD.value: Move.SHIELD,
             Move.RELOAD.value: Move.RELOAD,
             Move.SHOOT.value: Move.SHOOT,
             Move.SHOTGUN.value: Move.SHOTGUN,
             Move.ROCKET.value: Move.ROCKET}