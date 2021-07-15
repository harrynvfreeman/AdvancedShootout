import numpy as np
from mcts.mcts import Tree
from mcts.mcts import self_play as mcts_self_play
from game.agent.mcts_agent import MctsAgent
from game.env.advancedshootout_env import get_reward
import game.move
import gc
import json
import os
import copy
from game.utils import max_bullets, num_states, num_end_states, is_end_state

alpha = 0.1

class SavedState:
    def __init__(self, state, V, P):
        self.state = state
        self.V = V
        self.P = P
    
    def serialize(self, path):
        with open(path, 'w') as file:
            json.dump({'state':int(self.state), 'V':self.V.tolist(), 'P':self.P.tolist()}, file)
        
    def deserialize(self, path):
        with open(path, 'r') as file:
            j = json.load(file)
            self.state = j['state']
            self.V = np.array(j['V'])
            self.P = np.array(j['P'])
            

best_player_dir = "best"
self_play_data_dir = "data"

def initialize():
    #P = np.random.rand(num_states, game.move.num_moves)
    P = np.ones((num_states, game.move.num_moves))
    P = P + np.random.normal(loc=0, scale = 1, size=P.shape)*0.05
    P[P<0] = 0
    #we are not allowing reload if we have max bullets
    P[(max_bullets)*(max_bullets+1):, game.move.Move.RELOAD.value] = 0
    #cannot shoot with less than reload bullets
    P[0:((max_bullets+1)*game.move.move_bullet_cost[game.move.Move.SHOOT.value]), game.move.Move.SHOOT.value] = 0
    #cannot shotgun with less than shotgun bullets
    P[0:((max_bullets+1)*game.move.move_bullet_cost[game.move.Move.SHOTGUN.value]), game.move.Move.SHOTGUN.value] = 0
    #cannot rocket with less than rocket bullets
    P[0:((max_bullets+1)*game.move.move_bullet_cost[game.move.Move.ROCKET.value]), game.move.Move.ROCKET.value] = 0
    #cannot sonice boom with less than sonic boom bullets
    P[0:((max_bullets+1)*game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]), game.move.Move.SONIC_BOOM.value] = 0
    
    P = P / P.sum(axis=1, keepdims=True)
    P[-num_end_states:] = 0
    
    V = np.zeros((num_states))
    V = V + np.random.normal(loc=0, scale = 1, size=V.shape)*0.05
    
    #V = np.random.uniform(low=-1.0, high=1.0, size=num_states)
    #V = np.random.normal(loc=0, scale = 0.05, size=num_states)
    V[V<-1] = -1
    V[V>1] = 1
    
    os.system('rm -rf ./train/*')
    os.mkdir('./train/' + best_player_dir)
    os.mkdir('./train/0')
    os.mkdir('./train/' + self_play_data_dir)
    os.mkdir('./train/' + self_play_data_dir + '/0')
    
    os.mkdir('./train/' + best_player_dir + '/0')
    np.save('./train/'+ best_player_dir + '/0/P.npy', P)
    np.save('./train/'+ best_player_dir + '/0/V.npy', V)
    np.save('./train/' + best_player_dir + '/best_version.npy', 0)
    
    np.save('./train/0/P.npy', P)
    np.save('./train/0/V.npy', V)
    np.save('./train/version.npy', 0)
    
    os.mkdir('./train/opt_tmp')
    os.mkdir('./train/eval_tmp')
    
def optimize(version):
    V = np.load('./train/' + str(version) + '/V.npy')
    P = np.load('./train/' + str(version) + '/P.npy')
    
    visited = {}
    v_sum = np.zeros((num_states))
    p_sum = np.zeros((num_states, game.move.num_moves))
    
    saved_states = os.listdir('./train/' + self_play_data_dir + '/' + str(version))
    for saved_state_path in saved_states:
        path = './train/' + self_play_data_dir + '/' + str(version) + '/' + saved_state_path
        try:
            saved_state = SavedState(None, None, None)
            saved_state.deserialize(path)
            state = saved_state.state
        except:
            continue
        
        if visited.get(state) is None:
            visited[state] = 1
        else:
            visited[state] = visited[state] + 1
            
        v_sum[state] = v_sum[state] + saved_state.V
        p_sum[state, :] = p_sum[state, :] + saved_state.P[:]
        
    for state in visited:
        num_visit = visited[state]
        v_sum[state] = v_sum[state] / num_visit
        p_sum[state, :] = p_sum[state, :] / num_visit
        
        V[state] = V[state] + alpha*(v_sum[state] - V[state])
        P[state, :] = P[state, :] + alpha*(p_sum[state, :] - P[state, :])
        P[state, :] = P[state, :] / P[state, :].sum()
        
    np.save('./train/opt_tmp/V.npy', V)
    np.save('./train/opt_tmp/P.npy', P)
    
def evaluate(best_version, version, num_games=200, max_move_count=50, win_percent=0.55):
    challenger_path = './train/' + str(version)
    best_path = './train/' + best_player_dir + '/' + str(best_version)
    
    challenger_agent = MctsAgent(challenger_path, name="Challenger Agent")
    best_agent = MctsAgent(best_path, name="Best Agent")
    
    challenger_win_count = 0
    challenger_loss_count = 0
    draw_count = 0
    
    for game in range(num_games):
        print('Evaluating', game)
        challenger_agent.reset()
        best_agent.reset()
        
        game_done = False
        move_count = 0
        while move_count < max_move_count and not game_done:
            challenger_prev = copy.deepcopy(challenger_agent)
            best_prev = copy.deepcopy(best_agent)
            
            challenger_agent.make_action(challenger_agent.get_next_action(best_prev))
            best_agent.make_action(best_agent.get_next_action(challenger_prev))
            
            challenger_agent.post_move_update(challenger_agent.last_action, best_agent.last_action)
            best_agent.post_move_update(best_agent.last_action, challenger_agent.last_action)
            
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
        best_version = best_version + 1
        os.system('cp ./train/' + str(version) + '/P.npy ' './train/eval_tmp/P.npy')
        os.system('cp ./train/' + str(version) + '/V.npy ' './train/eval_tmp/V.npy')
        np.save('./train/eval_tmp/best_version.npy', best_version)
    else:
        np.save('./train/eval_tmp/best_version.npy', best_version)
    

def self_play(best_version, version, num_iterations=100):
    counter = 0
    
    P = np.load('./train/'+ best_player_dir + '/' + str(best_version) + '/P.npy')
    V = np.load('./train/'+ best_player_dir + '/' + str(best_version) + '/V.npy')
    
    for i in range(num_iterations):
        print('Beggining self play: ', i)
        counter = self_play_instance(V, P, counter, version)
        
    
def self_play_instance(V, P, counter, version):
    tree_a = Tree(V, P)
    tree_b = Tree(V, P)
    
    mcts_self_play(tree_a, tree_b, tree_a.max_moves)
    
    for i in range(int(tree_a.move_num + 1)):
        state = tree_a.states[i]
        V = tree_a.values[i]
        P = tree_a.policies[i, :]
        if (not is_end_state(state)) and (np.sum(P) == 0):
            raise Exception('Somehow added invalid policy')
        
        saved_state = SavedState(state, V, P)
        saved_state.serialize('./train/' + self_play_data_dir + '/' + str(version) + '/' + str(counter) + '_a.json')
        
        state = tree_b.states[i]
        V = tree_b.values[i]
        P = tree_b.policies[i, :]
        if (not is_end_state(state)) and (np.sum(P) == 0):
            raise Exception('Somehow added invalid policy')
        
        saved_state = SavedState(state, V, P)
        saved_state.serialize('./train/' + self_play_data_dir + '/' + str(version) + '/' + str(counter) + '_b.json')
        counter = counter + 1
    
    del tree_a
    del tree_b
    gc.collect()
    
    return counter
