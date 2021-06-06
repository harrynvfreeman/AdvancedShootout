import numpy as np
from scipy.optimize import linprog
'''
###WARNING This is the demo one

e = np.array([1, 1, 1])
A = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])

#want to maximize [x v]
#so want to minimize [-x, -v]
c = np.array([0, 0, 0, -1])

A_ub = np.zeros((3, 4))
A_ub[:, 0:3] = A
A_ub[:, 3] = e

A_eq = np.zeros((1,4))
A_eq[:, 0:3] = e

b_ub = np.array([0, 0, 0])
b_eq = np.array([1])

bounds = [(0, 1), (0, 1), (0, 1), (None, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

print(res.x)
'''

e = np.array([1, 1])
A = np.zeros((2,2))
A[0, 0] = 0.90222322
A[0,1] = 1.13498196
A[1, 0] = 0.59483798
A[1,1] = 0.89483798

#want to maximize [x v]
#so want to minimize [-x, -v]
c = np.array([0, 0, -1])

A_ub = np.zeros((2, 3))
A_ub[:, 0:2] = -A
A_ub[:, 2] = e

A_eq = np.zeros((1,3))
A_eq[:, 0:2] = e

b_ub = np.array([0, 0])
b_eq = np.array([1])

bounds = [(0, 1), (0, 1), (None, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
print(res.status)
print(res.x)