import game.move
from game.env.advancedshootout_env import get_reward

max_bullets = game.move.move_bullet_cost[game.move.Move.SONIC_BOOM.value]

num_move_states = (max_bullets+1)*(max_bullets+1)
num_end_states = 3
num_states =  num_move_states + num_end_states
win_state = num_move_states
loss_state = num_move_states + 1
draw_state = num_move_states + 2

def get_num_bullets(s):
    if is_end_state(s):
        raise Exception('Should not get num bullets when in end state')
    
    return s // (max_bullets+1)

def get_op_num_bullets(s):
    if is_end_state(s):
        raise Exception('Should not get op num bullets when in end state')
    
    return s % (max_bullets+1)

def get_state(num_bullets, op_num_bullets, reward):
    if reward == 1:
        return win_state
    if reward == -1:
        return loss_state
    
    return num_bullets * (max_bullets+1) + op_num_bullets

def invert_state(s):
    if is_end_state(s):
        if s == win_state:
            return loss_state
        if s == loss_state:
            return win_state
        return draw_state
    num_bullets = get_num_bullets(s)
    op_num_bullets = get_op_num_bullets(s)
    return get_state(op_num_bullets, num_bullets, 0)

def is_end_state(s):
    return s >= num_move_states

def is_legal_move(s, action):
    if is_end_state(s):
        return False
    num_bullets = get_num_bullets(s)
    if num_bullets == max_bullets and action.value == Move.RELOAD.value:
        return False
    return get_num_bullets(s) >= game.move.move_bullet_cost[action.value]

def get_next_state(s, action_a, action_b, player_a, player_b):
    player_a_bullets = get_num_bullets(s)
    player_b_bullets = get_op_num_bullets(s)
    
    player_a.force_num_bullets(player_a_bullets)
    player_b.force_num_bullets(player_b_bullets)
    
    player_a_bullets_next = player_a_bullets + player_a.get_bullet_diff(action_a)
    player_b_bullets_next = player_b_bullets + player_b.get_bullet_diff(action_b)
    
    reward = get_reward(action_a, action_b)
    
    next_state = get_state(player_a_bullets_next, player_b_bullets_next, reward)
    if next_state < 0:
        print("--------------")
        print(s)
        print('next state is ' + str(next_state) + ' from (' + str(player_a_bullets) + ', ' + str(player_b_bullets) + ') to (' + str(player_a_bullets_next) + ', ' + str(player_b_bullets_next) + ')')
        print('player a did ' + str(action_a) + ' and player b did ' + str(action_b))
        raise Exception("Illegal state")
    
    return next_state, reward