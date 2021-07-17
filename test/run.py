import node_test
import tree_test

#test.test_self_play_edges(1000, -1)

node_test.node_test_select_leaf_node_fail()
node_test.node_test_select_is_player_a()
node_test.node_test_select_is_player_b()
node_test.node_test_get_action_leaf_node_fail()
node_test.node_test_get_action_player_a_no_temp()
node_test.node_test_get_action_player_a_with_temp()

tree_test.tree_test_selet_expand_backup_root()
tree_test.tree_test_selet_expand_backup_twice()
tree_test.tree_test_selet_expand_backup_mult()

print('--------------')
tree_test.test_self_play_edges(1000)
print('--------------')
tree_test.test_self_play_edges(1000, root_child=0)
print('--------------')
tree_test.test_self_play_edges(1000, root_child=1)