import numpy as np
from quadproj import quadrics
from quadproj.project import project
from quasi_proj import quasi_proj
import matplotlib.pyplot as plt
#%%
d = 2
A = np.eye(d)
A[-1,-1] = 0

b = np.zeros(d)
b[-1] = -1

c = -1

Q = quadrics.Quadric(A, b, c)

#%%
x = np.random.uniform(-3, 3, size=(20, 100, d))
xx = quasi_proj(x,Q)
#%%
plt.close('all')
fig, ax = plt.subplots(1,2)
for i,z in enumerate([xx.reshape(-1, d)]):
    ax[i].scatter(x[:,0], x[:,1])
    ax[i].scatter(z[:, 0], z[:, 1], color='r')
    ax[i].quiver(x[:,0], x[:,1], (z[:, 0]-x[:,0]), (z[:, 1]-x[:,1]),
               scale=1., scale_units='xy', angles='xy',
               width=0.001)
