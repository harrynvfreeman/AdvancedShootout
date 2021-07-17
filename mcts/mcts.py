from game.move import num_moves
from game.move import move_dict
from game.agent.agent import Agent
import numpy as np
from game.utils import max_bullets, is_end_state, invert_state, is_legal_move, get_next_state, draw_state
import gc

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
        
    def select_edge(self, is_player_a, cpuct=12):
        if self.is_leaf:
            raise Exception("Cannot select on leaf node")
        
        if is_end_state(self.state):
            raise Exception("Cannot select on end state")
        
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
            
        
        N_total = np.sum(N)
        
        #we are adding 1 to N_total because we will try parent node count
        U = cpuct*P*np.sqrt(N_total + 1)/(1 + N)
        #U = cpuct*P*np.sqrt(N_total)/(1 + N)

        to_max = Q + U
        
        action_index = np.random.choice(np.flatnonzero(np.isclose(to_max, to_max.max())))   #np.argmax(Q+U)
        
        if not is_legal_move(state, move_dict[action_index]):
            raise Exception('Illegal action selected in select edge')
        return edges[action_index]
    
    def get_action(self, is_player_a, use_temp, temp):
        if self.is_leaf:
            raise Exception("Cannot call play on leaf node")
        
        if is_end_state(self.state):
            raise Exception("Cannot call play on end state")
        
        if is_player_a:
            edges = self.a_edges
        else:
            edges = self.b_edges
        
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
    def __init__(self, V, P, initial_state=0, move_thresh=100, max_moves=50, temp=1, num_sim=200):
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
        self.values = np.zeros((self.max_moves + 1))
        self.policies = np.zeros((self.max_moves + 1, num_moves))
        
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
            if not is_end_state(node.state):
                raise Exception("Reward is not 0 and end state not detected")
            return v_a, v_b
            #return reward, -reward
        
        if is_end_state(node.state):
            raise Exception('End state detected and reward not 0')
        
        #dirichlet noise if root and not expanded before
        if node.id == self.root.id:
            p_a, p_b = add_dirichlet_noise(node, p_a, p_b)
        
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
        
        if self.can_play:
            raise Exception("Not allowed to update")
        self.can_play = True
        
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
        
        if self.root.a_edges[0] is not None:
            if self.root.b_edges[0] is None:
                raise Exception('Why is a_edges none and not b_edges')
            p_a = np.zeros((num_moves))
            p_b = np.zeros((num_moves))
            for i in range(num_moves):
                p_a[i] = self.root.a_edges[i].P
                p_b[i] = self.root.b_edges[i].P
            p_a, p_b = add_dirichlet_noise(self.root, p_a, p_b)
            for i in range(num_moves):
                self.root.a_edges[i].P = p_a[i]
                self.root.b_edges[i].P = p_b[i]
        elif self.root.b_edges[0] is not None:
            raise Exception('Why is b_edges none and not a_edges')
        
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
        if not self.can_play:
            raise Exception("Not allowed to play")
        self.can_play = False
        
        for s in range(self.num_sim):
            node, reward = self.select()
            v_a, v_b = self.expand_and_evaluate(node, reward)
            self.backup(node, v_a, v_b)
        
        action, pi = self.play()
        return action, pi
    
def self_play(tree_a, tree_b, max_moves):
    for i in range(max_moves + 2):
        action_a, pi_a = tree_a.play_instance_get_move()
        action_b, pi_b = tree_b.play_instance_get_move()
        
        reward_a = tree_a.update_state(action_a, action_b, pi_a)
        reward_b = tree_b.update_state(action_b, action_a, pi_b)
        
        if reward_a != -reward_b:
            raise Exception('Rewards do not match')
        
        if tree_a.move_num != tree_b.move_num:
            raise Exception('Move nums do not match')
        
        if reward_a == 0 and tree_a.move_num != max_moves:
            continue
        
        tree_a.values[:] = reward_a
        tree_b.values[:] = reward_b
        
        if reward_a == 0:
            tree_a.states[tree_a.move_num] = draw_state
            tree_b.states[tree_a.move_num] = draw_state
        else:
            tree_a.states[tree_a.move_num] = tree_a.root.state
            tree_b.states[tree_b.move_num] = tree_b.root.state
            
        return
    raise Exception('Something went horribly wrong')

#dirichlet noise
def add_dirichlet_noise(node, p_a, p_b):
    dirichlet_a = np.random.dirichlet([0.5]*num_moves)
    dirichlet_b = np.random.dirichlet([0.5]*num_moves)
    
    for i in range(num_moves):
        if is_legal_move(node.state, move_dict[i]):
            p_a[i] = (1-0.25)*p_a[i] + (0.25)*dirichlet_a[i]
        if is_legal_move(invert_state(node.state), move_dict[i]):
            p_b[i] = (1-0.25)*p_b[i] + (0.25)*dirichlet_b[i]
            
    p_a = p_a / p_a.sum()
    p_b = p_b / p_b.sum()
    return p_a, p_b