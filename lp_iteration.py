import numpy as np
import game.move
from game.agent.agent import Agent
from game.env.advancedshootout_env import get_reward
import os
import game.lp_solve

max_bullets = 20

def get_num_bullets(s):
    return s // max_bullets

def get_op_num_bullets(s):
    return s % max_bullets

def get_state(num_bullets, op_num_bullets):
    return num_bullets * max_bullets + op_num_bullets

def run():
    num_states = max_bullets*max_bullets
    V = np.zeros((num_states))
    P = np.ones((num_states, game.move.num_moves))
    #we are not allowing reload if we have max bullets
    P[(max_bullets-1)*max_bullets:, game.move.Move.RELOAD.value] = 0
    #cannot shoot with 0 bullets
    P[0:max_bullets, game.move.Move.SHOOT.value] = 0
    #cannot shotgun with less than 2 bullets
    P[0:2*max_bullets, game.move.Move.SHOTGUN.value] = 0
    #cannot shotgun with less than 4 bulletsf
    P[0:4*max_bullets, game.move.Move.ROCKET.value] = 0
    
    P = P / P.sum(axis=1, keepdims=True)
    
    if os.path.isfile('./train/version.npy'):
        version_number = np.load('./train/version.npy') 
        current_P_path = './train/' + str(version_number) + '/P.npy'
        P_op = np.load(current_P_path)
        version_number = version_number + 1
    else:
        version_number = 0
        P_op = P.copy()
    
    
    agent = Agent("Agent")
    opponent = Agent("Opponent")
    
    discount_factor = .99
    
    #iterations = 20
    iterations = 5
    
    for iteration in range(iterations):
        V_previous = V.copy()
        for s in range(num_states):
            num_bullets = get_num_bullets(s)
            op_num_bullets = get_op_num_bullets(s)
            
            agent.force_num_bullets(num_bullets)
            opponent.force_num_bullets(op_num_bullets)
            
            valid_actions = agent.get_valid_actions()
            if num_bullets == max_bullets - 1:
                valid_actions[game.move.Move.RELOAD.value] = 0
            
            op_s = get_state(op_num_bullets, num_bullets)
            op_probs = P_op[op_s]
            
            #A = np.zeros((game.move.num_moves, game.move.num_moves))
            #invalid_rows = set()
            #invalid_cols = set()
            
            max_action_val = -np.inf
            max_action = None
            for i in range(valid_actions.shape[0]):
                if valid_actions[i] == 0:
                    #invalid_cols.add(i) 
                    continue
                
                a = game.move.move_dict[i]
                
                sum_v = 0
                
                num_bullets_next = num_bullets + agent.get_bullet_diff(a)
                
                for j in range(op_probs.shape[0]):
                    if op_probs[j] == 0:
                        #invalid_rows.add(j)
                        continue
                    
                    op_a = game.move.move_dict[j]
                    
                    op_num_bullets_next = op_num_bullets + opponent.get_bullet_diff(op_a)
                    
                    s_next = get_state(num_bullets_next, op_num_bullets_next)
                    
                    R = get_reward(a, op_a)
                    
                    if R == 0:
                        sum_v = sum_v + op_probs[j] * (R + discount_factor * V_previous[s_next])
                    else:
                        sum_v = sum_v + op_probs[j] * (R)
                    
                    #A[j, i] = (R + discount_factor * np.max(Q_previous[s_next]))
                if (sum_v > max_action_val):
                    max_action_val = sum_v
                    max_action = a
            
            V[s] = max_action_val
            #if iteration == iterations - 1:
            #    P[s, :] = 0
            #    P[s, max_action.value] = 1
    
            
            #if iteration == iterations - 1:
                #print(A)
            #    A = np.delete(A, list(invalid_rows), 0)
            #    A = np.delete(A, list(invalid_cols), 1)
                #print(A)
            #    res = game.lp_solve.solve(A)
            #    res_index = 0
            #    for i in range(valid_actions.shape[0]):
            #        if i not in invalid_cols:
            #            P[s, i] = res[res_index]
            #            res_index = res_index + 1
                #print(res)
                #print(P[s])
        
    P = np.zeros(P.shape)
    for s in range(num_states):
        num_bullets = get_num_bullets(s)
        op_num_bullets = get_op_num_bullets(s)
        
        agent.force_num_bullets(num_bullets)
        opponent.force_num_bullets(op_num_bullets)
        
        valid_actions = agent.get_valid_actions()
        if num_bullets == max_bullets - 1:
            valid_actions[game.move.Move.RELOAD.value] = 0
            
        op_s = get_state(op_num_bullets, num_bullets)
        op_probs = P_op[op_s]
        
        A = np.zeros((game.move.num_moves, game.move.num_moves))
        invalid_cols = set()
        invalid_rows = set()
        
        for i in range(valid_actions.shape[0]):
            if valid_actions[i] == 0:
                invalid_cols.add(i) 
                continue
            
            a = game.move.move_dict[i]
            num_bullets_next = num_bullets + agent.get_bullet_diff(a)
            
            for j in range(op_probs.shape[0]):
                if op_probs[j] == 0:
                    invalid_rows.add(j)
                    continue
                op_a = game.move.move_dict[j]
                op_num_bullets_next = op_num_bullets + opponent.get_bullet_diff(op_a)
                s_next = get_state(num_bullets_next, op_num_bullets_next)
                R = get_reward(a, op_a)
                if R == 0:
                    A[j, i] = V[s_next]
                else:
                    A[j, i] = R
                
        
        #A = np.delete(A, list(invalid_rows), 0)
        #y = np.delete(op_probs, list(invalid_rows))
        
        A = np.delete(A, list(invalid_cols), 1)
        
        A = np.delete(A, list(invalid_rows), 0)
        res = game.lp_solve.solve2(op_probs, A)
        
        #res = game.lp_solve.solve(op_probs, A)
        res_index = 0
        #print('')
        #print(s)
        #print(A)
        #print(res)
        for i in range(valid_actions.shape[0]):
            if i not in invalid_cols:
                P[s, i] = res[res_index]
                res_index = res_index + 1
        
    if not os.path.exists('./train/' + str(version_number)):
        os.mkdir('./train/' + str(version_number))
    np.save('./train/'+ str(version_number) + '/V.npy', V)
    np.save('./train/'+ str(version_number) + '/P.npy', P)
    np.save('./train/' + str(version_number) + '/max_bullets.npy', max_bullets)
    np.save('./train/version.npy', version_number)