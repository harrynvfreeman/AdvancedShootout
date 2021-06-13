import numpy as np
import game.move
from game.agent.agent import Agent
from game.agent.random_agent import RandomAgent
from game.env.advancedshootout_env import get_reward
import os

max_bullets = 20
def get_num_bullets(s):
    return s // max_bullets

def get_op_num_bullets(s):
    return s % max_bullets

def get_state(num_bullets, op_num_bullets):
    return num_bullets * max_bullets + op_num_bullets

num_states = max_bullets*max_bullets
V = np.zeros((num_states))
P = np.zeros((num_states, game.move.num_moves))

agent = Agent("Agent")
random_agent = Agent("RandomAgent")

discount_factor = .99
iterations = 20
for iteration in range(iterations):
    V_previous = V.copy()
    for s in range(num_states):
        num_bullets = get_num_bullets(s)
        op_num_bullets = get_op_num_bullets(s)
        
        agent.force_num_bullets(num_bullets)
        random_agent.force_num_bullets(op_num_bullets)
        
        valid_actions = agent.get_valid_actions()
        if num_bullets == max_bullets - 1:
            valid_actions[game.move.Move.RELOAD.value] = 0
        
        op_valid_actions = random_agent.get_valid_actions()
        if op_num_bullets == max_bullets - 1:
            op_valid_actions[game.move.Move.RELOAD.value] = 0
        prob = 1.0 / np.sum(op_valid_actions)
        
        max_action = None
        max_action_val = -np.inf
        for i in range(valid_actions.shape[0]):
            if valid_actions[i] == 0:
                continue
            
            a = game.move.move_dict[i]
            
            sum_v = 0
            
            num_bullets_next = num_bullets + agent.get_bullet_diff(a)
            
            for j in range(op_valid_actions.shape[0]):
                if op_valid_actions[j] == 0:
                    continue
                
                op_a = game.move.move_dict[j]
                
                op_num_bullets_next = op_num_bullets + random_agent.get_bullet_diff(op_a)
                
                s_next = get_state(num_bullets_next, op_num_bullets_next)
                
                R = get_reward(a, op_a)
                
                if R == 0:
                    sum_v = sum_v + prob * (R + discount_factor * V_previous[s_next])
                else:
                    sum_v = sum_v + prob * (R)
                    
            if sum_v > max_action_val:
                max_action_val = sum_v
                max_action = a
        V[s] = max_action_val
        if iteration == iterations - 1:
            P[s, max_action.value] = 1
        
version_number = 0
if not os.path.exists('./train/' + str(version_number)):
    os.mkdir('./train/' + str(version_number))
np.save('./train/'+ str(version_number) + '/V.npy', V)
np.save('./train/'+ str(version_number) + '/P.npy', P)
np.save('./train/' + str(version_number) + '/max_bullets.npy', max_bullets)
np.save('./train/version.npy', version_number)
