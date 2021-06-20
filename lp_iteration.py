import numpy as np
import game.move
from game.agent.agent import Agent
from game.agent.smart_agent import SmartAgent
from game.env.advancedshootout_env import get_reward
import game.lp_solve
import os
import copy

max_bullets = game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]
num_states = max_bullets*max_bullets
best_player_dir = "best"
challenger_dir = "challenger"
op_select_num = 1
num_iterations = 50
std = 0.04

def get_num_bullets(s):
    return s // max_bullets

def get_op_num_bullets(s):
    return s % max_bullets

def get_state(num_bullets, op_num_bullets):
    return num_bullets * max_bullets + op_num_bullets

#initiliaze for self play
#create P matrix with random values and save it as best player
def initialize():
    os.system('rm -rf ./train/*')
    os.mkdir('./train/' + best_player_dir)
    os.mkdir('./train/0')
    os.mkdir('./train/' + challenger_dir)
    
    P_best = np.random.rand(num_states, game.move.num_moves)
    #we are not allowing reload if we have max bullets
    P_best[(max_bullets-1)*max_bullets:, game.move.Move.RELOAD.value] = 0
    #cannot shoot with less than reload bullets
    P_best[0:(max_bullets*game.move.move_bullet_cost[game.move.Move.SHOOT.value]), game.move.Move.SHOOT.value] = 0
    #cannot shotgun with less than shotgun bullets
    P_best[0:(max_bullets*game.move.move_bullet_cost[game.move.Move.SHOTGUN.value]), game.move.Move.SHOTGUN.value] = 0
    #cannot rocket with less than rocket bullets
    P_best[0:(max_bullets*game.move.move_bullet_cost[game.move.Move.ROCKET.value]), game.move.Move.ROCKET.value] = 0
    #cannot sonice boom with less than sonic boom bullets
    P_best[0:(max_bullets*game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]), game.move.Move.SONIC_BOOM.value] = 0
    
    P_best = P_best / P_best.sum(axis=1, keepdims=True)
    
    if True:
        P_best = np.zeros((num_states, game.move.num_moves))
        #reload less than 2 bullets
        P_best[0:(max_bullets*2), game.move.Move.RELOAD.value] = 1
        #reload if op has 0 bullets and have less than 4
        #otherwise shield if less than 4
        P_best[(max_bullets*2), game.move.Move.RELOAD.value] = 1
        P_best[(max_bullets*2)+1:(max_bullets*3), game.move.Move.SHIELD.value] = 1
        P_best[(max_bullets*3), game.move.Move.RELOAD.value] = 1
        P_best[(max_bullets*3)+1:(max_bullets*4), game.move.Move.SHIELD.value] = 1
        #otherwise, rocket
        P_best[(max_bullets*4):, game.move.Move.ROCKET.value] = 1
    
    np.save('./train/'+ best_player_dir + '/P.npy', P_best)
    np.save('./train/0/P.npy', P_best)
    np.save('./train/version.npy', 0)
    

def evaluate(num_games, max_move_count, win_percent):
    #version_number = np.load('./train/version.npy')
    #challenger_P_path = './train/' + str(version_number) + '/P.npy'
    challenger_P_path = './train/' + challenger_dir + '/P.npy'
    
    best_P_path = './train/' + best_player_dir + '/P.npy'
    
    challenger_agent = SmartAgent(challenger_P_path, deterministic=False, name="Challenger Agent")
    best_agent = SmartAgent(best_P_path, deterministic=False, name="Best Agent")
    
    challenger_win_count = 0
    challenger_loss_count = 0
    draw_count = 0
    
    for game in range(num_games):
        challenger_agent.reset()
        best_agent.reset()
        
        game_done = False
        move_count = 0
        while move_count < max_move_count and not game_done:
            challenger_prev = copy.deepcopy(challenger_agent)
            best_prev = copy.deepcopy(best_agent)
            
            challenger_agent.make_action(challenger_agent.get_next_action(best_prev))
            best_agent.make_action(best_agent.get_next_action(challenger_prev))
            reward = get_reward(challenger_agent.last_action, best_agent.last_action)
            game_done = (reward != 0)
            move_count = move_count + 1
            
        if reward == 1:
            challenger_win_count = challenger_win_count + 1
        elif reward == -1:
            challenger_loss_count = challenger_loss_count + 1
        else:
            draw_count = draw_count + 1
    
    print('Win rate is: ' + str(challenger_win_count / num_games))
    print('Loss rate is: ' + str(challenger_loss_count / num_games))
    print('Draw rate is: ' + str(draw_count / num_games))
    if challenger_win_count / num_games >= win_percent:
        os.system('cp ./train/' + challenger_dir + '/P.npy ' './train/' + best_player_dir + '/P.npy')
        version_number = np.load('./train/version.npy') + 1
        os.mkdir('./train/' + str(version_number))
        os.system('cp ./train/' + challenger_dir + '/P.npy ' './train/' + str(version_number) + '/P.npy')
        np.save('./train/version.npy', version_number)
        return True
    else:
        return False

def self_play(num_updates, alpha):
    #select checkpoint to play against
    version_number = np.load('./train/version.npy')
    op_version = np.random.randint(max(0, version_number - op_select_num + 1), high=version_number + 1)
    op_P_path = './train/' + str(op_version) + '/P.npy'
    P = np.load(op_P_path)
    
    P_orig = copy.deepcopy(P)
    
    for i in range(num_updates):
        P = update(P, alpha)
        
    #version_number = version_number + 1
    #os.mkdir('./train/' + str(version_number))
    #np.save('./train/'+ str(version_number) + '/P.npy', P)
    #np.save('./train/version.npy', version_number)
    np.save('./train/' + challenger_dir + '/P.npy', P)
    print(np.max(abs(P - P_orig)))
    #return np.max(abs(P - P_op))
    
#add noise?
def update(P_op, alpha):
    agent = Agent("Agent")
    opponent = Agent("Opponent")
    
    #discount_factor = 1 we don't need this
    
    #P_rand = np.random.normal(0, scale=std, size=P_op.shape)
    #P_rand[P_op < 0.001] = 0
    #P_op = P_op + P_rand
    #P_op[P_op < 0] = 0
    #P_op[P_op > 1] = 1
    #P_op = P_op / P_op.sum(axis=1, keepdims=True)
    
    V = np.zeros((num_states))
    for iteration in range(num_iterations):
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
            
            max_action_val = -np.inf
            max_action = None
            for i in range(valid_actions.shape[0]):
                if valid_actions[i] == 0:
                    continue
                
                a = game.move.move_dict[i]
                sum_v = 0
                num_bullets_next = num_bullets + agent.get_bullet_diff(a)
                
                for j in range(op_probs.shape[0]):
                    if op_probs[j] == 0:
                        continue
                    
                    op_a = game.move.move_dict[j]
                    op_num_bullets_next = op_num_bullets + opponent.get_bullet_diff(op_a)
                    s_next = get_state(num_bullets_next, op_num_bullets_next)
                    R = get_reward(a, op_a)
                    
                    if R == 0:
                        sum_v = sum_v + op_probs[j] * (V_previous[s_next])
                    else:
                        sum_v = sum_v + op_probs[j] * (R)
                        
                if (sum_v > max_action_val):
                    max_action_val = sum_v
                    max_action = a
            
            V[s] = max(min(max_action_val, 1), -1)
            
    P_new = np.zeros(P_op.shape)
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
                    
        A = np.delete(A, list(invalid_cols), 1)
        A = np.delete(A, list(invalid_rows), 0)
        
        try:
            res = game.lp_solve.solve(A)
        except:
            print('Error caught lp solve')
            return P_op
        
        res_index = 0
        for i in range(valid_actions.shape[0]):
            if i not in invalid_cols:
                P_new[s, i] = res[res_index]
                res_index = res_index + 1
                
    P = alpha*P_new + (1-alpha)*P_op #P_op + (P_new - P_op)*alpha
    P = P / P.sum(axis=1, keepdims=True)
    
    
    return P