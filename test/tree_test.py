import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from mcts.mcts import Tree, Node, Edge
from game.move import num_moves, move_dict, Move

### Class Tree Tests
def tree_test_selet_expand_backup_root():
    V = np.load('./V.npy')
    P = np.load('./P.npy')
    v_comp = 1.23
    V[0] = v_comp
    tree = Tree(V, P)
    
    node, reward = tree.select()
    
    if node.id != 0:
        raise Exception('tree_test_selet_expand_backup_root test failed. Node id is not 0')
    
    if not node.is_leaf:
        raise Exception('tree_test_selet_expand_backup_root test failed. Node is not leaf')
    
    if tree.node_id_counter != 1:
        raise Exception('tree_test_selet_expand_backup_root test failed. Node id counter is not 1')
    
    if reward != 0:
        raise Exception('tree_test_selet_expand_backup_root test failed. Reward is not 0')
    
    for i in range(num_moves*num_moves):
        if node.children[i] is not None:
            raise Exception('tree_test_selet_expand_backup_root test failed. children not None for root select')
    
    for i in range(num_moves):
        if node.a_edges[i] is not None:
            raise Exception('tree_test_selet_expand_backup_root test failed. a_edge is not None')
        if node.b_edges[i] is not None:
            raise Exception('tree_test_selet_expand_backup_root test failed. b_edge is not None')
        
    v_a, v_b = tree.expand_and_evaluate(node, reward)
    
    if v_a != v_comp:
        raise Exception('tree_test_selet_expand_backup_root test failed. Unexpected v_a')
    
    if v_b != v_comp:
        raise Exception('tree_test_selet_expand_backup_root test failed. Unexpected v_b')
    
    if node.is_leaf:
        raise Exception('tree_test_selet_expand_backup_root test failed. Node is still leaf')
    
    P_a_sum = 0
    P_b_sum = 0
    for i in range(num_moves):
        if node.a_edges[i] is None:
            raise Exception('tree_test_selet_expand_backup_root test failed. a_edge is None')
        P_a_sum = P_a_sum + node.a_edges[i].P
        if node.b_edges[i] is None:
            raise Exception('tree_test_selet_expand_backup_root test failed. b_edge is None')
        P_b_sum = P_b_sum + node.b_edges[i].P
    
    if not np.isclose(P_a_sum, 1):
        raise Exception('tree_test_selet_expand_backup_root test failed. P_a_sum is not 1')
    
    if not np.isclose(P_b_sum, 1):
        raise Exception('tree_test_selet_expand_backup_root test failed. P_b_sum is not 1')
    
    tree.backup(node, v_a, v_b)
    
    for i in range(num_moves):
        if node.a_edges[i].N != 0:
            raise Exception('tree_test_selet_expand_backup_root test failed. node a_edges N is not 0')
        if node.b_edges[i].N != 0:
            raise Exception('tree_test_selet_expand_backup_root test failed. node b_edges N is not 0')

def tree_test_selet_expand_backup_twice():
    V = np.load('./V.npy')
    P = np.load('./P.npy')
    v_comp = 1.27
    V[0] = v_comp
    V[1] = v_comp
    V[11] = v_comp
    V[12] = v_comp
    
    tree = Tree(V, P)
    
    for i in range(2):
        node, reward = tree.select()
        v_a, v_b = tree.expand_and_evaluate(node, reward)
        tree.backup(node, v_a, v_b)
        
    if tree.node_id_counter != 2:
        raise Exception('tree_test_selet_expand_backup_twice test failed. Node id counter is not 3')
    
    V_sum = 0
    for i in range(num_moves):
        V_sum = V_sum + tree.root.a_edges[i].W
    
    if not np.isclose(V_sum, v_comp):
        raise Exception('tree_test_selet_expand_backup_root test failed. V_sum is not v_comp. Expected: ', v_comp, ', Actual: ', V_sum)
    
def tree_test_selet_expand_backup_mult():
    V = np.load('./V.npy')
    P = np.load('./P.npy')
    v_comp = 1.27
    V[:] = v_comp
    
    tree = Tree(V, P)
    
    to_sub = 0
    reward_set = set()
    num_itr = 100
    for i in range(num_itr):
        node, reward = tree.select()
        v_a, v_b = tree.expand_and_evaluate(node, reward)
        tree.backup(node, v_a, v_b)
        
        if reward != 0:
            if node.id in reward_set:
                to_sub = to_sub + 1
            else:
                reward_set.add(node.id)
    
    id_exp = num_itr - to_sub    
    if tree.node_id_counter != id_exp:
        raise Exception('tree_test_selet_expand_backup_mult test failed. Node id counter is not expected.  Expected: ', id_exp, ', Actual: ', tree.node_id_counter)
    
    V_sum = 0
    N_sum = 0
    Q_sum = 0
    for i in range(num_moves):
        V_sum = V_sum + tree.root.a_edges[i].W
        N_sum = N_sum + tree.root.a_edges[i].N
        Q_sum = Q_sum + tree.root.a_edges[i].Q * tree.root.a_edges[i].N
        
    
    if not np.isclose(V_sum, v_comp*(num_itr-1)):
        raise Exception('tree_test_selet_expand_backup_mult test failed. V_sum is not expected. Expected: ', v_comp*(num_itr-1), ', Actual: ', V_sum)
    
    if not np.isclose(Q_sum, v_comp*(num_itr-1)):
        raise Exception('tree_test_selet_expand_backup_mult test failed. Q_sum is not expected. Expected: ', v_comp*(num_itr-1), ', Actual: ', Q_sum)
    
    if N_sum != (num_itr-1):
        raise Exception('tree_test_selet_expand_backup_mult test failed. N_sum is not expected. Expected: ', num_itr-1, ', Actual: ', N_sum)
    
    
def test_self_play_edges(num_select, root_child=-1):
    V = np.load('./V.npy')
    P = np.load('./P.npy')
    
    tree = Tree(V, P)
    
    for s in range(num_select):
            node, reward = tree.select()
            v_a, v_b = tree.expand_and_evaluate(node, reward)
            tree.backup(node, v_a, v_b)
    
    if root_child == -1:
        node = tree.root
    else:
        node = tree.root.children[root_child]
        if node is None:
            print('State was unvisited.  Re-run test.')
    
    for edge in node.a_edges:
        print(edge.action, ': ', edge.parent_node.state, ', ', edge.N, ', ', edge.W, ', ', edge.Q, ', ', edge.P)