import lp_iteration

max_iter = 100
decay_iter = 5
decay_val = 0.9
diff_thresh = 0.005
alpha = 0.1

iteration = 0
while iteration < max_iter:
    if iteration > 0 and iteration % decay_iter == 0:
        alpha = decay_val*alpha
        print('New alpha is: ' + str(alpha))
    max_diff = lp_iteration.run(iteration, alpha)
    print('Done iteration: ' + str(iteration))
    print('Max diff: ' + str(max_diff))
    if max_diff < diff_thresh:
        print('Done')
        break
    print('')
    iteration = iteration + 1