import numpy as np
import game.move
#doesn't have to be random agent, I should have a base class
from game.random_agent import RandomAgent

max_bullets = 20
def get_num_bullets(s):
    return s // max_bullets

def get_op_num_bullets(s):
    return s % max_bullets

def get_valid_actions(s):
    #no bullets
    if get_num_bullets(s) == 0:
        return [game.move.Move.SHIELD, game.move.Move.RELOAD]
    if get_num_bullets(s) == max_bullets - 1:
        return [game.move.Move.SHIELD, game.move.Move.SHOOT]
    return [game.move.Move.SHIELD, game.move.Move.RELOAD, game.move.Move.SHOOT]

def get_op_valid_actions(s):
    #no bullets
    if get_op_num_bullets(s) == 0:
        return [game.move.Move.SHIELD, game.move.Move.RELOAD]
    if get_op_num_bullets(s) == max_bullets - 1:
        return [game.move.Move.SHIELD, game.move.Move.SHOOT]
    return [game.move.Move.SHIELD, game.move.Move.RELOAD, game.move.Move.SHOOT]

def get_state(num_bullets, op_num_bullets):
    return num_bullets * max_bullets + op_num_bullets

num_states = max_bullets*max_bullets
Q = np.zeros((num_states, game.move.num_moves))
#cannot shoot if we have zero bullets
Q[0:max_bullets, game.move.Move.SHOOT.value] = -np.inf
#we are not allowing reload if we have max bullets
Q[(max_bullets-1)*max_bullets:, game.move.Move.RELOAD.value] = -np.inf

agent = RandomAgent("Agent")
random_agent = RandomAgent()


discount_factor = .9

iterations = 50
for i in range(iterations):
    Q_previous = Q.copy()
    for s in range(num_states):
        num_bullets = get_num_bullets(s)
        op_num_bullets = get_op_num_bullets(s)
        for a in get_valid_actions(s):
            sum_v = 0
            
            if a == game.move.Move.RELOAD:
                num_bullets_next = num_bullets + 1
            elif a == game.move.Move.SHOOT:
                num_bullets_next = num_bullets - 1
            else:
                num_bullets_next = num_bullets
            
            op_actions = get_op_valid_actions(s)
            prob = 1.0 / len(op_actions)
            for op_a in op_actions:
                if op_a == game.move.Move.RELOAD:
                    op_num_bullets_next = op_num_bullets + 1
                elif op_a == game.move.Move.SHOOT:
                    op_num_bullets_next = op_num_bullets - 1
                else:
                    op_num_bullets_next = op_num_bullets
                
                s_next = get_state(num_bullets_next, op_num_bullets_next)
                
                if a == game.move.Move.SHOOT and op_a == game.move.Move.RELOAD:
                    R = 1
                elif a == game.move.Move.RELOAD and op_a == game.move.Move.SHOOT:
                    R = -1
                else:
                    R = 0
                    
                sum_v = sum_v + prob * (R + discount_factor * np.max(Q_previous[s_next]))
                
            Q[s, a.value] = sum_v
            
np.save('Q.npy', Q)
