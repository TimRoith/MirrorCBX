from cbx.dynamics import CBO
import numpy as np

class signedDistance:
    def get_proj_matrix(self, v):
        grad = self.grad(v)
        outer = np.einsum('...i,...j->...ij',grad,grad)
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


    
    
class SphereCBO(CBO):
    def __init__(self, f, sdf = None, **kwargs):
        super().__init__(f, **kwargs)
        self.sdf = get_sdf(sdf)
        
    def inner_step(self,):
        self.compute_consensus() # compute consensus, sets self.energy and self.consensus
        self.drift = self.x - self.consensus
        Px = self.sdf.get_proj_matrix(self.x) # compute projection of x
        
        # compute relevant terms for update
        dir_drift = self.x + self.dt * apply_proj(Px, self.consensus)
        noise     = self.sigma * apply_proj(Px, self.noise())
        constr    = (
            self.dt * self.sigma**2/2 * self.sdf.Laplacian(self.x) *
            self.drift**2 * self.sdf.grad(self.x)
        )
        
        x_tilde = dir_drift + noise + constr
        self.x = self.sdf.proj(x_tilde)