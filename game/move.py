from enum import Enum

num_moves = 6

class Move(Enum):
    SHIELD = 0
    RELOAD = 1
    SHOOT = 2
    SHOTGUN = 3
    ROCKET = 4
    SONIC_BOOM = 5
    
move_dict = {Move.SHIELD.value: Move.SHIELD,
             Move.RELOAD.value: Move.RELOAD,
             Move.SHOOT.value: Move.SHOOT,
             Move.SHOTGUN.value: Move.SHOTGUN,
             Move.ROCKET.value: Move.ROCKET,
             Move.SONIC_BOOM.value: Move.SONIC_BOOM}

move_bullet_cost = {Move.SHIELD.value: 0,
             Move.RELOAD.value: 0,
             Move.SHOOT.value: 1,
             Move.SHOTGUN.value: 2,
             Move.ROCKET.value: 4,
             Move.SONIC_BOOM.value: 10}

move_bullet_gain = {Move.SHIELD.value: 0,
             Move.RELOAD.value: 1,
             Move.SHOOT.value: 0,
             Move.SHOTGUN.value: 0,
             Move.ROCKET.value: 0,
             Move.SONIC_BOOM.value: 0}