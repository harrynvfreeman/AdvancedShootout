import lp_iteration

max_iter = 100
num_updates = 5
alpha = .1
eval_games = 1000
max_move_count = 50
win_percent = 0.55

#lp_iteration.initialize()
#iteration = 0
#while iteration < max_iter:
#    print('Starting iteration: ' + str(iteration))
#    lp_iteration.self_play(num_updates, alpha)
#    if lp_iteration.evaluate(eval_games, max_move_count, win_percent):
#        print('Best player updated')
#        
#    print('Done iteration: ' + str(iteration))
#    print('')
#    iteration = iteration + 1
#


import mcts_iteration

#mcts_iteration.initialize()

#mcts_iteration.optimize()
#if mcts_iteration.evaluate():
#    print('Best player updated')

#for i in range(100):
#    print("Starting cycle ", i)
#    mcts_iteration.self_play()
i=0
while True:
    print('Optimizing ' + str(i))
    mcts_iteration.optimize()
    print('Evaluating ' + str(i))
    i = i + 1
    if mcts_iteration.evaluate():
        print('Best player updated')
