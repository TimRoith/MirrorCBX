from mirrorcbx.dynamics import MirrorCBO
from cbx.objectives import Rastrigin, Ackley
from cbx.scheduler import multiply
import numpy as np

class scaled_Ackley:
    def __init__(self, l= 1, **kwargs):
        self.f = Ackley(**kwargs)
        self.l = np.atleast_1d(l)
        
    def __call__(self, x):
        return self.f(self.l[None,None,:] *x)
        

#%%
class cov_mm:
    def __init__(self, C=None, d=1):
        self.C = 1 if C is None else C
    def grad(self, x):
        return self.C *  x
    def grad_conj(self, y):
        # if isinstance(self.C, np.ndarray):
        #     #return np.einsum('kij,klj->kli', self.C, y)
        #     return np.linalg.solve(self.C, y[..., None])[..., 0]
        # else:
        #     return y
        return (1/(self.C))**0.5 * y
    
def post_process_cov(dyn):
    # alpha = dyn.alpha.copy()
    # dyn.alpha*=0
    # dyn.update_covariance()
    # dyn.alpha = alpha
    #dyn.mirrormap.C += dyn.Cov_sqrt#[:, np.arange(dyn.d[0]), np.arange(dyn.d[0])][:,None,:]
    d = dyn.x_old - dyn.x
    ll = .5
    #dyn.mirrormap.C = ll * dyn.mirrormap.C + (1 - ll)* (d * d).mean(axis=1)
    dyn.mirrormap.C += 0.1*(d * d).mean(axis=1)
    dyn.x = np.clip(dyn.x, a_min=-1e8, a_max=1e8)
    
#%%
d = 5
x = np.random.uniform(-1,1, size=(1,50, d))
mm = cov_mm(C = 0.001)

f = scaled_Ackley(minimum=np.random.uniform(-1,1,d)[None,None,:])

A = np.diag(np.random.uniform(0.1,5,d))
b = np.random.uniform(-1,1, d)

class mmA:
    def __init__(self, A):
        self.A = A
    def grad(self, x):
        return self.A * x
    def grad_conj(self, x):
        return (1/self.A) * x

def f(x):
    return ((x@A.T) * x).sum(axis=-1) + (b*x).sum(axis=-1)

dyn = MirrorCBO(
    f, x=x,
    alpha=1, dt=.001, 
    noise='anisotropic',
    sigma=2.,
    max_it=2500,
    #mirrormap=mmA(np.diag(A)),
    track_args={'names':['x', 'y']}
    #post_process= post_process_cov
)
dyn.optimize(sched=multiply(factor=1.05, maximum=1e18))