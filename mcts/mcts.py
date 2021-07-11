from game.move import num_moves
from game.move import move_dict
from game.move import move_bullet_cost
from game.move import Move
from game.agent.agent import Agent
from game.env.advancedshootout_env import get_reward
import numpy as np
import gc
import pickle

max_bullets = move_bullet_cost[Move.SONIC_BOOM.value]

def get_num_bullets(s):
    return s // (max_bullets+1)

def get_op_num_bullets(s):
    return s % (max_bullets+1)

def get_state(num_bullets, op_num_bullets):
    return num_bullets * (max_bullets+1) + op_num_bullets

def invert_state(s):
    num_bullets = get_num_bullets(s)
    op_num_bullets = get_op_num_bullets(s)
    return get_state(op_num_bullets, num_bullets)

def is_legal_move(s, action):
    num_bullets = get_num_bullets(s)
    if num_bullets == max_bullets and action.value == Move.RELOAD.value:
        return False
    return get_num_bullets(s) >= move_bullet_cost[action.value]

def get_next_state(s, action_a, action_b, player_a, player_b):
    player_a_bullets = get_num_bullets(s)
    player_b_bullets = get_op_num_bullets(s)
    
    player_a.force_num_bullets(player_a_bullets)
    player_b.force_num_bullets(player_b_bullets)
    
    player_a_bullets_next = player_a_bullets + player_a.get_bullet_diff(action_a)
    player_b_bullets_next = player_b_bullets + player_b.get_bullet_diff(action_b)
    
    next_state = get_state(player_a_bullets_next, player_b_bullets_next)
    if next_state < 0:
        print("--------------")
        print(s)
        print('next state is ' + str(next_state) + ' from (' + str(player_a_bullets) + ', ' + str(player_b_bullets) + ') to (' + str(player_a_bullets_next) + ', ' + str(player_b_bullets_next) + ')')
        print('player a did ' + str(action_a) + ' and player b did ' + str(action_b))
        raise Exception("Illegal state")
    reward = get_reward(action_a, action_b)
    return next_state, reward

class Node:
    def __init__(self, state, id, parent=None):
        #integer number representing game state
        self.state = state
        #id for uniqueness
        self.id = id
        #game moves
        self.a_edges = [None]*num_moves
        self.b_edges = [None]*num_moves
        #next states
        self.children = [None]*num_moves*num_moves
        self.children_visit = np.zeros((num_moves*num_moves))
        #parent
        self.parent = parent
        #not expanded / evaluated
        self.is_leaf = True
        #depth for debugging
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
            
        #for backup
        self.parent_a_edge = None
        self.parent_b_edge = None        
        
    def select_edge(self, is_player_a, cpuct=10):
        if self.is_leaf:
            raise Exception("Cannot select on leaf node")
        
        Q = np.zeros((num_moves))
        P = np.zeros((num_moves))
        N = np.zeros((num_moves))
        
        if is_player_a:
            edges = self.a_edges
            state = self.state
        else:
            edges = self.b_edges
            state = invert_state(self.state)
        
        for i in range(num_moves):
            if is_legal_move(state, move_dict[i]):
                Q[i] = edges[i].Q
            else:
                Q[i] = -np.inf
                if P[i] != 0:
                    raise Exception('P for illegal move is not 0')
                if N[i] != 0:
                    raise Exception('N for illegal move is not 0')
            P[i] = edges[i].P
            N[i] = edges[i].N
        
        #if not player a just do numpy random choice
        if not is_player_a:
            action_index = np.random.choice(num_moves, p=P)
            if not is_legal_move(state, move_dict[action_index]):
                raise Exception('Illegal action selected in select edge')
            return edges[action_index]
            
        
        N_total = np.sum(N)
        
        #we are adding 1 to N_total because we will try parent node count
        U = cpuct*P*np.sqrt(N_total + 1)/(1 + N)

        to_max = Q + U
        
        action_index = np.random.choice(np.flatnonzero(np.isclose(to_max, to_max.max())))   #np.argmax(Q+U)
        
        if not is_legal_move(state, move_dict[action_index]):
            raise Exception('Illegal action selected in select edge')
        return edges[action_index]
    
    def get_action(self, is_player_a, use_temp, temp):
        if self.is_leaf:
            raise Exception("Cannot call play on leaf node")
        
        if is_player_a:
            edges = self.a_edges
        else:
            edges = self.b_edges
            
        #if not player a then random based on probabilities
        if not is_player_a:
            P = np.zeros((num_moves))
            for i in range(num_moves):
                P[i] = edges[i].P
            action_index = np.random.choice(num_moves, p=P)
            return move_dict[action_index], None
        
        N = np.zeros((num_moves))
        for i in range(num_moves):
            N[i] = edges[i].N
            if use_temp:
                N[i] = np.power(N[i], 1/temp)
        
        N_total = np.sum(N)
        
        pi = N / N_total
        if use_temp:
            return move_dict[np.random.choice(num_moves, p=pi)], pi
        else:
            arg_max = np.random.choice(np.flatnonzero(np.isclose(N, N.max()))) #np.argmax(N)
            return move_dict[arg_max], pi
        
class Edge:
    def __init__(self, parent_node, p, action):
        self.parent_node = parent_node
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = p
        self.action = action

class Tree:
    def __init__(self, V, P, initial_state=0, move_thresh=100, max_moves=50, temp=1, num_sim=400):
        self.V = V
        self.P = P
        self.move_thresh = move_thresh
        self.move_num = 0
        self.max_moves = max_moves
        self.temp = 1
        self.num_sim = num_sim
        
        self.node_id_counter = 0
        self.root = Node(initial_state, self.node_id_counter)
        self.node_id_counter = self.node_id_counter + 1
        
        self.player_a = Agent()
        self.player_b = Agent()
        
        self.states = np.zeros((self.max_moves) + 1, dtype=np.int)
        self.values = np.zeros((self.max_moves))
        self.policies = np.zeros((self.max_moves, num_moves))
        
        self.can_play = True
        
    def select(self):
        #start at root node
        node = self.root
        reward = 0
        #select until leaf node is reached
        while not node.is_leaf:
            edge_a = node.select_edge(True)
            edge_b = node.select_edge(False)
            
            node.children_visit[edge_a.action.value + num_moves*edge_b.action.value] = node.children_visit[edge_a.action.value + num_moves*edge_b.action.value] + 1
            
            next_state, reward = get_next_state(node.state, edge_a.action, edge_b.action, self.player_a, self.player_b)
            
            if node.children[edge_a.action.value + num_moves*edge_b.action.value] is None:
                new_node = Node(next_state, self.node_id_counter, parent=node)
                self.node_id_counter = self.node_id_counter + 1
                node.children[edge_a.action.value + num_moves*edge_b.action.value] = new_node
                node = new_node
            else:
                node = node.children[edge_a.action.value + num_moves*edge_b.action.value]
            node.parent_a_edge = edge_a
            node.parent_b_edge = edge_b
        return node, reward
    
    def expand_and_evaluate(self, node, reward):
        if not node.is_leaf:
            raise Exception("Cannot call expand and evaluate on non-leaf node")
        
        v_a = np.array(self.V[node.state])
        p_a = np.array(self.P[node.state])
        
        inverted_state = invert_state(node.state)
        v_b = np.array(self.V[inverted_state])
        p_b = np.array(self.P[inverted_state])
        
        #if game over we do not expand or evaluate
        if reward != 0:
            return v_a, v_b
            #return reward, -reward
        
        #dirichlet noise if root
        if node.id == self.root.id:
            dirichlet_a = np.random.dirichlet([0.03]*num_moves)
            dirichlet_b = np.random.dirichlet([0.03]*num_moves)
            
            for i in range(num_moves):
                if is_legal_move(node.state, move_dict[i]):
                    p_a[i] = (1-0.25)*p_a[i] + (0.25)*dirichlet_a[i]
                if is_legal_move(inverted_state, move_dict[i]):
                    p_b[i] = (1-0.25)*p_b[i] + (0.25)*dirichlet_b[i]
                    
            p_a = p_a / p_a.sum()
            p_b = p_b / p_b.sum()
        
        for i in range(num_moves):
            node.a_edges[i] = Edge(node, p_a[i], move_dict[i])
            node.b_edges[i] = Edge(node, p_b[i], move_dict[i])
            
        node.is_leaf = False;
        return v_a, v_b
    
    def backup(self, node, v_a, v_b):
        while node.id != self.root.id:
            edge_a = node.parent_a_edge            
            edge_b = node.parent_b_edge
            
            edge_a.N = edge_a.N + 1
            edge_a.W = edge_a.W + v_a
            edge_a.Q = edge_a.W / edge_a.N
            
            edge_b.N = edge_b.N + 1
            edge_b.W = edge_b.W + v_b
            edge_b.Q = edge_b.W / edge_b.N
            
            node.parent_a_edge = None
            node.parent_b_edge = None
            
            if edge_a.parent_node.state != edge_b.parent_node.state:
                raise Exception("Edges parent nodes do not match")
            node = node.parent
        if node.parent is not None:
            raise Exception('Last node backup parent is not none')
    
    def play(self):
        if not self.can_play:
            raise Exception("Not allowed to play")
        self.can_play = False
        
        node = self.root
        
        if self.move_num <= self.move_thresh:
            use_temp = True
            temp = self.temp
        else:
            use_temp = False
            temp = None
        
        action_a, pi = node.get_action(True, use_temp, temp)
        return action_a, pi
    
    def update_state(self, action_a, action_b, pi):
        node = self.root
        
        if self.move_num <= self.move_thresh:
            use_temp = True
            temp = self.temp
        else:
            use_temp = False
            temp = None
        
        if self.can_play:
            raise Exception("Not allowed to update")
        self.can_play = True
        
        if action_b is None:
            action_b, _ = node.get_action(False, use_temp, temp)
        
        next_state, reward = get_next_state(node.state, action_a, action_b, self.player_a, self.player_b)
        
        if node.children[action_a.value + num_moves*action_b.value] is None:
            node.children[action_a.value + num_moves*action_b.value] = Node(next_state, self.node_id_counter, parent=node)
            self.node_id_counter = self.node_id_counter + 1
        else:
            if node.children[action_a.value + num_moves*action_b.value].parent_a_edge is not None:
                raise Exception("Why is parent a edge not none?")
            if node.children[action_a.value + num_moves*action_b.value].parent_b_edge is not None:
                raise Exception("Why is parent b edge not none?")
        
        self.root = node.children[action_a.value + num_moves*action_b.value]
        self.root.parent = None
        
        if pi is not None:
            self.states[self.move_num] = node.state
            self.policies[self.move_num,:] = pi[:]
            self.move_num = self.move_num + 1
        
        #garbage collect
        for edge in node.a_edges:
            del edge
        for edge in node.b_edges:
            del edge
        del node
        gc.collect()
        
        return reward
        
    def play_instance_get_move(self):
        for s in range(self.num_sim):
            node, reward = self.select()
            v_a, v_b = self.expand_and_evaluate(node, reward)
            self.backup(node, v_a, v_b)
        
        action, pi = self.play()
        return action, pi
        
    def self_play_instance(self):
        action, pi = self.play_instance_get_move()
        reward = self.update_state(action, None, pi)
        return reward
    
    def self_play(self):
        for i in range(self.max_moves + 2):
            reward = self.self_play_instance()
            if reward == 1:
                #print('Challenger won after ' + str(self.move_num) + ' moves')
                self.values[:] = reward
                self.states[self.move_num] = self.root.state
                return
                #return reward
            elif reward == -1:
                #print('Best won after ' + str(self.move_num) + ' moves')
                self.values[:] = reward
                self.states[self.move_num] = self.root.state
                return
                #return reward
            elif reward == 0 and self.move_num == self.max_moves:
                #print('Draw after ' + str(self.move_num) + ' moves')
                self.values[:] = reward
                self.states[self.move_num] = self.root.state
                return
                #return reward
        print(i)
        raise Exception('Something went horribly wrong')
        