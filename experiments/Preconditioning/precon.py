#%%
from mirrorcbx.dynamics import MirrorCBO
from cbx.dynamics.cbo import CBO, cbo_update
import matplotlib.pyplot as plt
import numpy as np

#%%
d = 2
A = np.diag(np.ones(d,))
A[-1, -1] = 5
b = 0

class mmA:
    def __init__(self, A):
        self.A = A
    def grad(self, x):
        return self.A * x
    def grad_conj(self, x):
        return (1/self.A) * x
    
class preconCBO_var2(CBO):
    def __init__(self, f, A, **kwargs):
        super().__init__(f, **kwargs)
        self.A = A
    
    def inner_step(self,):
        self.compute_consensus()
        self.drift = (self.x - self.consensus)
        self.x += cbo_update(
             1/(2*np.diag(A)) * self.drift, self.lamda, self.dt, 
            self.sigma, self.noise()
        )

def f(x):
    return ((x@A.T) * x).sum(axis=-1) + (b*x).sum(axis=-1)

# %% Gradient Descent and Preconditioned Gradient Descent
def gd(A, b, x0=None, max_it=100, dt=0.1):
    x = np.random.uniform(-1,1, d) if x0 is None else x0.copy()
    hs = []
    for i in range(max_it):
        x -= dt * (2 * x @ A.T + b)
        hs.append(f(x))
    return x, hs

def precon_gd(A, b, x0 = None, max_it=100, dt=0.1):
    x = np.random.uniform(-1,1, d) if x0 is None else x0.copy()
    hs = []
    for i in range(max_it):
        x -= dt * 1/(2*np.diag(A)) * (2 * x @ A.T + b)
        hs.append(f(x))
    return x, hs

#%%
num_dts = 21
dts = np.linspace(0.1, 2.1, num_dts)
es  = np.zeros((num_dts, 2))
esgd = np.zeros((num_dts, 2))

cbo_kwargs = {
    'alpha':10, 
    'noise':'isotropic', 'sigma':1.,
    'max_it':100, 'verbosity':0, 
    'track_args':{'names':[]}
}

x = np.random.uniform(-1,1, size=(30, 20, d))
for i, dt in enumerate(dts):
    for j, mm in enumerate([None, mmA(2 * np.diag(A))]):
        dyn = MirrorCBO(
            f, x=x.copy(), dt=dt,
            mirrormap=mm, **cbo_kwargs
        )
        dyn.optimize()
        es[i, j] = f(dyn.consensus).mean()
    # dyn = preconCBO_var2(f, A, x=x, dt=dt, **cbo_kwargs)
    # dyn.optimize()
    # es[i, -1] = f(dyn.consensus).mean()

    xgd, h = gd(A, b, x0=x[:,0,:], max_it=10, dt=dt)
    xpgd, hp = precon_gd(A, b, x0=x[:,0,:], max_it=10, dt=dt)

    esgd[i, 0] = f(xgd).mean()
    esgd[i, 1] = f(xpgd).mean()
        
    print('Finished for dt = ', dt)

#%%
for i in [0,1]: plt.loglog(dts, es[:, i], marker='*')
#for i in [0,1]: plt.plot(dts, esgd[:, i])

#%%
#np.savetxt('precon.csv', np.concatenate([dts[:, None], es, esgd], axis=1), delimiter=' ')