import numpy as np
#%%
name= 'CBO'
path0 = 'results/' + name
path1 = '_diff_c.txt'
num = 6
A = np.zeros((num, 2))
A[:, 0] = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
for i in range(num):
    A[i, 1] = np.loadtxt(path0 + str(i) + path1)[-1]
np.savetxt(path0+'sweep_res.txt', A)