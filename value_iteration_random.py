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
Q = np.zeros((num_states, game.move.num_moves))
#we are not allowing reload if we have max bullets
Q[(max_bullets-1)*max_bullets:, game.move.Move.RELOAD.value] = -np.inf
#cannot shoot with 0 bullets
Q[0:max_bullets, game.move.Move.SHOOT.value] = -np.inf
#cannot shotgun with less than 2 bullets
Q[0:2*max_bullets, game.move.Move.SHOTGUN.value] = -np.inf
#cannot shotgun with less than 4 bullets
Q[0:4*max_bullets, game.move.Move.ROCKET.value] = -np.inf

#cannot attack if we have zero bullets
#do we need this?
#Q[0:max_bullets, game.move.Move.SHOOT.value] = -np.inf
#we are not allowing reload if we have max bullets
#Q[(max_bullets-1)*max_bullets:, game.move.Move.RELOAD.value] = -np.inf

agent = Agent("Agent")
random_agent = Agent("RandomAgent")


discount_factor = .9

iterations = 20
for i in range(iterations):
    Q_previous = Q.copy()
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
                
                sum_v = sum_v + prob * (R + discount_factor * np.max(Q_previous[s_next]))
                
            Q[s, a.value] = sum_v
            
P = np.zeros(Q.shape)
for s in range(num_states):
    index = np.argmax(Q[s])
    P[s, index] = 1
version_number = 0
if not os.path.exists('./train/' + str(version_number)):
    os.mkdir('./train/' + str(version_number))
np.save('./train/'+ str(version_number) + '/Q.npy', Q)
np.save('./train/'+ str(version_number) + '/P.npy', P)
np.save('./train/' + str(version_number) + '/max_bullets.npy', max_bullets)
np.save('./train/version.npy', version_number)
