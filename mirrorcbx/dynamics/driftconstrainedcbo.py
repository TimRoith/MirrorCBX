from cbx.dynamics import CBO
import numpy as np
#%% Constraints

class MultiConstraint:
    def __init__(self, constraints):
        self.constraints = [] if constraints is None else constraints
        
    def grad_squared_sum(self, x):
        out = np.zeros_like(x)
        for con in self.constraints:
            out += con.grad_squared(x)
        return out
            
    def hessian_squared_sum(self, x):
        out = np.zeros(x.shape + (x.shape[-1],))
        for con in self.constraints:
            out += con.hessian_squared(x)
        return out
    
    
class Constraint:
    def grad_squared(self, x):
        return 2 * self(x)[..., None] * self.grad(x)
    
    def hessian_squared(self, x):
        grad = self.grad(x)
        outer = np.einsum('...i,...j->...ij', grad, grad)
        return 2 * (outer + self(x)[..., None, None] * self.hessian(x))
    
class NoConstraint(Constraint):
    def __init__(self,):
        super().__init__()
        
    def __call__(self, x):
        return np.zeros(x.shape[:-1])
    
    def grad(self, x):
        return np.zeros_like(x)
    
    def hessian(self, x):
        return np.zeros(x.shape + (x.shape[-1],))
    
    
class quadricConstraint(Constraint):
    def __init__(self, A = None, b = None, c = 0):
        self.A = A
        self.b = b
        self.c = c
        
    def __call__(self, x):
         return (
             (x * (x@self.A)) .sum(axis=-1) + 
             (x * self.b).sum(axis=-1) + 
             self.c
        )
        
    def grad(self, x):
        return 2 * x@self.A + self.b
    
    def hessian(self, x):
        return 2 * self.A
    
class sphereConstraint(Constraint):
    def __init__(self, r=1.):
        super().__init__()
        self.r = r
        
    def __call__(self, x):
        return np.linalg.norm(x, axis=-1)**2 - self.r
    
    def grad(self, x):
        return 2 * x
    
    def hessian(self, x):
        return 2 * np.tile(np.eye(x.shape[-1]), x.shape[:-1] + (1,1))
    

class planeConstraint(Constraint):
    def __init__(self, a=0, b=1.):
        super().__init__()
        self.a = a
        self.norm_a = np.linalg.norm(a, axis=-1)
        self.b = b
        
    def __call__(self, x):
        return ((self.a * x).sum(axis=-1) - self.b)/self.norm_a
    
    def grad(self, x):
        return (self.a/self.norm_a) * np.ones_like(x)
    
    def hessian(self, x):
        return np.zeros(x.shape + (x.shape[-1],))
    
    
const_dict = {'plane': planeConstraint, 
              'sphere': sphereConstraint,
              'quadric': quadricConstraint}    

def get_constraints(const):
    CS = []
    const = [] if const is None else const
    for c in const:
        if c is None:
            pass #return NoConstraint()
        else:
            CS.append(const_dict[c['name']](**{k:v for k,v in c.items() if k not in ['name']}))
    return CS


#%%
def solve_system(A, x):
    return np.linalg.solve(A, x[..., None])[..., 0]

#%%
def dc_inner_step(self,):
    self.compute_consensus()
    self.drift = self.x - self.consensus
    noise = self.sigma * self.noise()
    const_drift = (self.dt/ self.eps) * self.G.grad_squared_sum(self.x)
    scaled_drift = self.lamda * self.dt * self.drift
    
    x_tilde = scaled_drift + const_drift + noise
    A = np.eye(self.d[0]) + (self.dt/ self.eps) * self.G.hessian_squared_sum(self.x)
    self.x -= solve_system(A, x_tilde)


#%%
class DriftConstrainedCBO(CBO):
    '''
    Implements the algorithm in [1]
    
    
    [1] Carrillo, José A., et al. 
    "An interacting particle consensus method for constrained global 
    optimization." arXiv preprint arXiv:2405.00891 (2024).
    '''
    
    
    def __init__(
            self, f, 
            constraints = None, eps=0.01,  
            eps_indep=0.001, sigma_indep=0., 
            **kwargs
        ):
        super().__init__(f, **kwargs)
        self.G = MultiConstraint(get_constraints(constraints))
        self.eps = eps
        self.eps_indep = eps_indep
        self.sigma_indep = sigma_indep
        
        
    def inner_step(self,):
        self.indep_noise_step()
        dc_inner_step(self)
        
    def indep_noise_step(self,):
        if self.sigma_indep > 0 and (self.consensus is not None):
            while True:
                cxd = (np.linalg.norm(self.x - self.consensus, axis=-1)**2).mean(axis=-1)/self.d[0]
                idx = np.where(cxd < self.eps_indep)
                if len(idx[0]) == 0:
                    break
                z = np.random.normal(0,1, size=((len(idx[0]),) + self.x.shape[1:]))
                self.x[idx[0], ...] += (
                    self.sigma_indep * 
                    self.dt**0.5 * 
                    z
                )

