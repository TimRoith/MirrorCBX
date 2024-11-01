import numpy as np
from quadproj import quadrics
from quadproj.project import project

y = np.array([[ 0.62855944,  0.01257362, -0.24681933],
       [ 0.53383014, -0.0817122 , -0.15117589],
       [-0.52166377, -0.08310256, -0.09460424],
       [-0.52558052,  0.0088466 ,  0.10800053],
       [-0.50738795, -0.09807329,  0.06471006]])
dim=3
A = np.eye(dim)
A[0, 0] = 4
A[1, 1] = -2
A[2, 2] = -1
b = np.zeros(dim)
c = -1

Q = quadrics.Quadric(A, b, c)

#%%
project(Q, y[:3,:])