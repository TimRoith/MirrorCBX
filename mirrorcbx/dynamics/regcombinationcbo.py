from cbx.dynamics import CBO
import numpy as np
from mirrorcbx.constraints import get_constraints, MultiConstraint
from mirrorcbx.regularization import regularize_objective
#%%
def solve_system(A, x):
    return np.linalg.solve(A, x[..., None])[..., 0]

#%% regularized CBO ver 2 -- by Urbain et al.
class RegCombinationCBO(CBO):
    '''
    Implements the algorithm in [1]
    
    
    [1] Carrillo, José A., et al. 
    "Carrillo, J. A., Totzeck, C., & Vaes, U. (2023). 
    Consensus-based optimization and ensemble kalman inversion for 
    global optimization problems with constraints" 
    arXiv preprint https://arxiv.org/abs/2111.02970 (2024).
    '''

    def __init__(self, f, constraints = None, eps=0.01, nu = 1, **kwargs):
        super().__init__(f, **kwargs)
        self.G = MultiConstraint(get_constraints(constraints))
        self.eps = eps
        self.f = regularize_objective(self.f, self.G.squared, lamda=1/nu)
    
    def inner_step(self, ):
        self.compute_consensus()
        self.drift = self.x - self.consensus
        noise = self.sigma * self.noise()

        #  update particle positions
        x_tilde = self.x - self.lamda * self.dt * self.drift + noise
        A = (
            np.eye(self.d[0]) + 
            2 * (self.dt/ self.eps) * self.G.call_times_hessian(self.x)
        )

        # Step 3: Solve the system A * X = X for each (M, N) entry in self.x
        self.x = solve_system(A, x_tilde)


