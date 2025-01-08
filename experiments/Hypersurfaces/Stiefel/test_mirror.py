from mirrorcbx.dynamics import MirrorCBO
from cbx.objectives import Ackley
import numpy as np

#%%
class Stiefel_mirror:
    def __init__(self, nk=None):
        self.nk = (1,1) if nk is None else nk
        
    def grad(self, x):
        return x
    
    def grad_conj(self, y):
        yshape = y.shape
        y = y.reshape((-1,) + self.nk)
        U, _, Vh = np.linalg.svd(y, full_matrices=False)
        return (U@Vh).reshape(yshape)
    
#%%
(n,k) = (20,10)
x = np.random.uniform(-1,1, (5, 100, n*k))
mm = Stiefel_mirror((n,k))
xx = mm.grad_conj(x)

M = np.random.uniform(-1,1, (n,k))
U, _, Vh = np.linalg.svd(M, full_matrices=False)
M = (U@Vh).reshape((n*k))

f = Ackley(minimum=M)

dyn = MirrorCBO(
    f, x=x, noise='anisotropic',
    sigma=1., 
    mirrormap=mm,
    max_it=400,
    dt=0.1,
    alpha=1000)

dyn.optimize()