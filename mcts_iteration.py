import numpy as np
from mcts.mcts import Tree
from game.agent.smart_agent import SmartAgent
from game.env.advancedshootout_env import get_reward
import game.move
import gc
import pickle
import os
import copy

counter_max = 1000
alpha = 0.1

class SavedState:
    def __init__(self, state, V, P):
        self.state = state
        self.V = V
        self.P = P

best_player_dir = "best"
self_play_data_dir = "data"

max_bullets = game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]
num_states = (max_bullets+1)*(max_bullets+1)

def initialize():
    P = np.random.rand(num_states, game.move.num_moves)
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
    V = np.random.uniform(low=-1.0, high=1.0, size=num_states)
    
    os.system('rm -rf ./train/*')
    os.mkdir('./train/' + best_player_dir)
    os.mkdir('./train/0')
    os.mkdir('./train/' + self_play_data_dir)
    
    os.mkdir('./train/' + best_player_dir + '/0')
    np.save('./train/'+ best_player_dir + '/0/P.npy', P)
    np.save('./train/'+ best_player_dir + '/0/V.npy', V)
    np.save('./train/' + best_player_dir + '/best_version.npy', 0)
    
    np.save('./train/0/P.npy', P)
    np.save('./train/0/V.npy', V)
    np.save('./train/version.npy', 0)
    
    np.save('./train/count.npy', -1)

def optimize(batch_size=32, num_training_steps=100):
    version = np.load('./train/version.npy')
    V = np.load('./train/' + str(version) + '/V.npy')
    P = np.load('./train/' + str(version) + '/P.npy')
    for i in range(num_training_steps):
        saved_states = os.listdir('./train/' + self_play_data_dir)
        indeces = np.random.choice(len(saved_states), size=batch_size, replace=False)
        for index in indeces:
            path = './train/' + self_play_data_dir + '/' + saved_states[index]
            with open(path, 'rb') as handle:
                saved_state = pickle.load(handle)
            state = saved_state.state
            V[state] = (1-alpha)*V[state] + (alpha)*saved_state.V
            P[state, :] = (1-alpha)*P[state, :] + (alpha)*saved_state.P[:]
            
    version = version + 1
    os.mkdir('./train/' + str(version))
    np.save('./train/' + str(version) + '/V.npy', V)
    np.save('./train/' + str(version) + '/P.npy', P)
    np.save('./train/version.npy', version)
    
def evaluate(num_games=1000, max_move_count=20, win_percent=0.55):
    version = np.load('./train/version.npy')
    best_version = np.load('./train/' + best_player_dir + '/best_version.npy')
    challenger_P_path = './train/' + str(version) + '/P.npy'
    best_P_path = './train/' + best_player_dir + '/' + str(best_version) + '/P.npy'
    
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
    if True or challenger_win_count / num_games >= win_percent:
        best_version = best_version + 1
        os.mkdir('./train/' + best_player_dir + '/' + str(best_version))
        os.system('cp ./train/' + str(version) + '/P.npy ' './train/' + best_player_dir + '/' + str(best_version) + '/P.npy')
        os.system('cp ./train/' + str(version) + '/V.npy ' './train/' + best_player_dir + '/' + str(best_version) + '/V.npy')
        np.save('./train/' + best_player_dir + '/best_version.npy', best_version)
        return True
    else:
        return False
    

def self_play(num_iterations=10):
    best_version = np.load('./train/' + best_player_dir + '/best_version.npy')
    P = np.load('./train/'+ best_player_dir + '/' + str(best_version) + '/P.npy')
    V = np.load('./train/'+ best_player_dir + '/' + str(best_version) + '/V.npy')
    
    for i in range(num_iterations):
        print('--------------------', i)
        print('Beggining self play: ', i)
        self_play_instance(V, P)
        #saved_states = self_play_instance(V, P)
        #with open('./train/' + self_play_data_dir + '/' + str(i) + '.pickle', 'wb') as handle:
        #    pickle.dump(saved_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
def self_play_instance(V, P):
    tree = Tree(V, P)
    tree.self_play()
    
    states = tree.states
    values = tree.values
    policies = tree.policies
    
    #saved_states = [None]*tree.move_num
    counter = np.load('./train/count.npy')
    if counter < 0  or counter >= counter_max:
        counter = 0
        
    for i in range(int(tree.move_num)):
        state = tree.states[i]
        V = tree.values[i]
        P = tree.policies[i, :]
        
        saved_state = SavedState(state, V, P)
        with open('./train/' + self_play_data_dir + '/' + str(counter) + '.pickle', 'wb') as handle:
            pickle.dump(saved_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        counter = counter + 1
        #saved_states[i] = saved_state
    
    np.save('./train/count.npy', counter)
    
    del tree
    gc.collect()
    
    #return saved_states