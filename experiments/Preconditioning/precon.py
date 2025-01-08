from mirrorcbx.dynamics import MirrorCBO
import matplotlib.pyplot as plt
import numpy as np

#%%
A = np.diag([1,5])
b = 0

class mmA:
    def __init__(self, A):
        self.A = A
    def grad(self, x):
        return self.A * x
    def grad_conj(self, x):
        return (1/self.A) * x

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
d = 2

num_dts = 21
dts = np.linspace(0.1, 2.1, num_dts)
es  = np.zeros((num_dts, 2))
esgd = np.zeros((num_dts, 2))

x = np.random.uniform(-1,1, size=(50, 20, d))
for i, dt in enumerate(dts):
    for j, mm in enumerate([None, mmA(2 * np.diag(A))]):
        dyn = MirrorCBO(
            f, x=x,
            alpha=10, dt=dt, 
            noise='isotropic',
            sigma=.1,
            max_it=100,
            mirrormap=mm,
            verbosity=0,
            track_args={'names':[]},
            #post_process= post_process_cov
        )
        dyn.optimize()
        es[i, j] = f(dyn.consensus).mean()

    xgd, h = gd(A, b, x0=x[:,0,:], max_it=10, dt=dt)
    xpgd, hp = precon_gd(A, b, x0=x[:,0,:], max_it=10, dt=dt)

    esgd[i, 0] = f(xgd).mean()
    esgd[i, 1] = f(xpgd).mean()
        
    print('Finished for dt = ', dt)

for i in [0,1]: plt.plot(dts, es[:, i])
for i in [0,1]: plt.plot(dts, esgd[:, i])

#%%
np.savetxt('precon.csv', np.concatenate([dts[:, None], es, esgd], axis=1), delimiter=' ')