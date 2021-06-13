import game.move
import numpy as np
from scipy.optimize import linprog

def solve(A, debug=False):
    m = A.shape[0]
    n = A.shape[1]
    
    #e = np.ones((m))
    
    c = np.zeros((n+1))
    c[-1] = -1
    
    A_ub = np.zeros((m, n+1))
    A_ub[:, 0:n] = -A
    A_ub[:, n] = 1
    
    A_eq = np.zeros((1,n+1))
    A_eq[:, 0:n] = 1
    
    b_ub = np.zeros((m))
    b_eq = np.array([1])

    bounds = [None]*(n+1)
    for i in range(n):
        bounds[i] = (0,None)
        #bounds[i] = (0,1)
    bounds[n] = (None,None)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, options={'sym_pos':False})
    
    if res.status > 1:
        raise Exception("Res status is: " + str(res.status))
    
    if res.status == 1:
        print("WARNING WARNING WARNING Res status is 1")
    
    return res.x[0:n]