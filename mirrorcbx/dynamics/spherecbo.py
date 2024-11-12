from cbx.dynamics import CBO
import numpy as np
from scipy.special import logsumexp as logsumexp_scp

#%%
class signedDistance:
    def get_proj_matrix(self, v):
        grad = self.grad(v)
        #outer = np.einsum('...i,...j->...ij',grad,grad)
        outer = grad[..., None,:]*grad[..., None]
        return np.eye(v.shape[-1]) - outer
    
class sphereDistance(signedDistance):
    def grad(self, v):
        return v/np.linalg.norm(v,axis=-1, keepdims=True)
    
    def Laplacian(self, v):
        d = v.shape[-1]
        return (d-1) * np.linalg.norm(v, axis=-1, keepdims=True)
    
    def proj(self, x):
        return x/np.linalg.norm(x, axis=-1, keepdims=True)
    
class planeDistance(signedDistance):
    def __init__(self, a=0, b=1.):
        self.a = a
        self.norm_a = np.linalg.norm(a, axis=-1)
        self.b = b
        
    def grad(self, v):
        return self.a/self.norm_a * np.ones_like(v)
    
    def Laplacian(self, v):
        return 0
    
    def proj(self, y):
        return y - ((self.a * y).sum(axis=-1, keepdims=True) - self.b)/(self.norm_a**2) * self.a

        
sdf_dict = {'sphere': sphereDistance, 'plane': planeDistance}    

def get_sdf(sdf):
    if sdf is None:
        return sphereDistance()
    else:
        return sdf_dict[sdf['name']](**{k:v for k,v in sdf.items() if k not in ['name']})

def apply_proj(P, z):
    return np.einsum('...ij,...j->...i',P, z)

def compute_consensus_rescale(energy, x, alpha):
    emin = np.min(energy, axis=-1, keepdims=True)
    weights = - alpha * (energy - emin)
    coeff_expan = tuple([Ellipsis] + [None for i in range(x.ndim-2)])
    coeffs = np.exp(weights - logsumexp_scp(weights, axis=-1, keepdims=True))[coeff_expan]

    return (x * coeffs).sum(axis=1, keepdims=True), energy


    
    
class SphereCBO(CBO):
    def __init__(self, f, sdf = None, **kwargs):
        super().__init__(
            f, 
            compute_consensus = compute_consensus_rescale, 
            **kwargs
        )
        self.sdf = get_sdf(sdf)
        
    def inner_step(self,):
        self.compute_consensus() # compute consensus, sets self.energy and self.consensus
        self.drift = self.x - self.consensus
        Px = self.sdf.get_proj_matrix(self.x) # compute projection of x
        
        # compute relevant terms for update
        # dir_drift = self.x + self.dt * apply_proj(Px, self.consensus)
        # noise     = self.sigma * apply_proj(Px, self.noise())
        
        # perform addition before matrix multiplaction to save time
        drift_and_noise = (
            self.x + 
            apply_proj(Px, 
            self.dt * self.consensus + self.sigma * self.noise())
        )
        
        # note that in the implementation here:
        # https://github.com/PhilippeSu/KV-CBO/blob/master/v2/kvcbo/KuramotoIteration.m
        # norm(drift)**2 is used instead of drift**2
        constr  = (
            self.dt * self.sigma**2/2 * self.sdf.Laplacian(self.x) *
            np.linalg.norm(self.drift, axis=-1, keepdims=True)**2 * 
            #self.drift**2 *
            self.sdf.grad(self.x)
        )
        
        x_tilde = drift_and_noise - constr
        self.x = self.sdf.proj(x_tilde)
        
#     class sigma_sched:
#         def __init__(self, eta=2., tau=1.2):
#             self.eta = eta
#             self.tau = tau
            
#         def update(self, dyn):
#             if np.std(dyn.x, axis=-2).sum() < self.eta:
#                 dyn.sigma /= self.tau
            
#     sched=multiply(factor=1.01, maximum=1e18)