import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from mcts.mcts import Tree, Node, Edge
from game.move import num_moves, move_dict, Move

### Class Node Tests
def node_test_select_leaf_node_fail():
    node = Node(0, 0)
    succ = False
    try:
        action = node.select_edge(True)
    except Exception as err:
        if (str(err)) == "Cannot select on leaf node":
            succ = True
        else:
            raise err
    if not succ:
        raise Exception('Succ test failed in node_test_select_leaf_node_fail')
    
#warning warning warning - do I need to test not P = 1 and other types?
#perhaps a test for illegal move?
def node_test_select_is_player_a():
    node = Node(0, 0)
    node.is_leaf = False
    for i in range(num_moves):
        node.a_edges[i] = Edge(node, 0, move_dict[i])
        
    #set shield to 100% probability
    node.a_edges[0].P = 1
    
    #make sure no probabilities
    for i in range(100):
        edge = node.select_edge(True)
    
        if edge.action.value != Move.SHIELD.value:
            raise Exception('node_test_is_player_a test failed. Expected: ', 0, ', Actual: ', edge.action.value)
        
#warning warning warning - do I need to test not P = 1 and other types?
#perhaps a test for illegal move?
def node_test_select_is_player_b():
    node = Node(0, 0)
    node.is_leaf = False
    for i in range(num_moves):
        node.b_edges[i] = Edge(node, 0, move_dict[i])
        
    #set shield to 100% probability
    node.b_edges[0].P = 1
    
    #make sure no probabilities
    for i in range(100):
        edge = node.select_edge(False)
    
        if edge.action.value != Move.SHIELD.value:
            raise Exception('node_test_is_player_b test failed. Expected: ', 0, ', Actual: ', edge.action.value)
    
def node_test_get_action_leaf_node_fail():
    node = Node(0, 0)
    succ = False
    try:
        action = node.get_action(True, False, None)
    except Exception as err:
        if (str(err)) == "Cannot call play on leaf node":
            succ = True
        else:
            raise err
            
    if not succ:
        raise Exception('Succ test failed in node_test_get_action_leaf_node_fail')
    
def node_test_get_action_player_a_no_temp():
    node = Node(0, 48)
    node.is_leaf = False
    for i in range(num_moves):
        node.a_edges[i] = Edge(node, 0, move_dict[i])
        
    #set shield to 100% probability
    node.a_edges[0].N = 10
    node.a_edges[1].N = 20
    node.a_edges[2].N = 40
    node.a_edges[3].N = 10
    node.a_edges[4].N = 20
    
    for i in range(100):
        action, pi = node.get_action(True, False, None)
    
        expected =  np.array([0.1, 0.2, 0.4, 0.1, 0.2, 0])
        if not (pi == expected).all():
            raise Exception('node_test_get_action_player_a_no_temp test failed. Expected: ', expected, ', Actual: ', pi)
        
        if not action.value == Move.SHOOT.value:
            raise Exception('node_test_get_action_player_a_no_temp test failed. Expected: ', Move.SHOOT, ', Actual: ', action)
    
def node_test_get_action_player_a_with_temp():
    node = Node(0, 48)
    node.is_leaf = False
    for i in range(num_moves):
        node.a_edges[i] = Edge(node, 0, move_dict[i])
        
    #set shield to 100% probability
    node.a_edges[0].N = 10
    node.a_edges[1].N = 20
    node.a_edges[2].N = 40
    node.a_edges[3].N = 10
    node.a_edges[4].N = 20
    
    found_max = False
    found_not_max = False
    for i in range(100):
        move, pi = node.get_action(True, True, 1)
        
        if move.value == Move.SHOOT.value:
            found_max = True
        else:
            found_not_max = True
        
        expected =  np.array([0.1, 0.2, 0.4, 0.1, 0.2, 0])
        if not (pi == expected).all():
            raise Exception('node_test_get_action_player_a_with_temp test failed. Expected: ', expected, ', Actual: ', pi)
    
    if not found_max:
        raise Exception('Best move was never selected in node_test_get_action_player_a_with_temp')
    
    if not found_not_max:
        raise Exception('Best move was always selected in node_test_get_action_player_a_with_temp')