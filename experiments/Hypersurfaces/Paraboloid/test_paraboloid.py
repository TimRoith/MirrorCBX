import numpy as np
from quadproj import quadrics
from quadproj.project import project

#%%
d = 2
A = np.eye(d)
A[-1,-1] = 0

b = np.zeros(d)
b[-1] = -1

c = -1

Q = quadrics.Quadric(A, b, c)

#%%
x = np.random.uniform(-2,2, (100, d))

xx = project(Q, x)

#%%
import matplotlib.pyplot as plt

d = xx - x

for z in [x,xx]:
    plt.scatter(z[:,0], z[:,1])
plt.quiver(x[:,0], x[:,1], d[:,0], d[:,1],
           **{'scale':1, 'scale_units':'xy', 'angles':'xy', 
           'width':0.005, 'color':'r'})