from .pdyn_const import Constrained_CBXDynamic
import numpy as np

#%% regularized CBO ver 3 -- by Yuhua et al.
class regCBO3(Constrained_CBXDynamic):
    r"""Consensus-based optimization (CBO) class

    This class implements the CBO algorithm as described in [1]_. The algorithm
    is a particle dynamic algorithm that is used to minimize the objective function :math:`f(x)`.

    Parameters
    ----------
    x : array_like, shape (N, d)
        The initial positions of the particles. For a system of :math:`N` particles, the i-th row of this array ``x[i,:]``
        represents the position :math:`x_i` of the i-th particle.
    f : objective
        The objective function :math:`f(x)` of the system.
    dt : float, optional
        The parameter :math:`dt` of the system. The default is 0.1.
    lamda : float, optional
        The decay parameter :math:`\lambda` of the system. The default is 1.0.
    alpha : float, optional
        The heat parameter :math:`\alpha` of the system. The default is 1.0.
    noise : noise_model, optional
        The noise model that is used to compute the noise vector. The default is ``normal_noise(dt=0.1)``.
    sigma : float, optional
        The parameter :math:`\sigma` of the noise model. The default is 1.0.
    
    References
    ----------
    .. [1] Pinnau, R., Totzeck, C., Tse, O., & Martin, S. (2017). A consensus-based model for global optimization and its mean-field limit. 
        Mathematical Models and Methods in Applied Sciences, 27(01), 183-204.

    """

    def __init__(self, f, **kwargs) -> None:
        super().__init__(f, **kwargs)
        
    
    def inner_step(self, global_min) -> None:
        r"""Performs one step of the CBO algorithm.

        Parameters
        ----------
        None

        Returns
        -------
        None
        
        """
        # update, consensus point, drift and energy
        self.consensus, energy = self.compute_consensus()
        self.energy[self.consensus_idx] = energy
        self.drift = self.x[self.particle_idx] - self.consensus # consensus = (M, 1, d), x = (M, N, d)
        
        # compute noise
        self.s = self.sigma * self.noise()

        # compute temporary variables
        self.temp = self.lamda * self.dt * self.drift + self.s
        
        # update particle positions by solving the linear system using np.linalg.solve 
        gd_g = self.grad_eq_constraint # (M, N, d)
        gd_gsqr = 2*np.multiply(self.eq_constraint[:, :, None], self.grad_eq_constraint) # (M, N, d) 
        self.temp = (self.dt / self.epsilon) * gd_gsqr + self.temp # (M, N, d) = dim of self.x

        Hess_g = self.hess_eq_constraint # (M, N, d, d) -> use (d, d)
        for i in range(self.M):
            for j in range(self.N):
                gd_g_vec = np.reshape(gd_g[i, j, :], ((gd_g.shape)[-1], 1)) 
                Hess_gsqr = 2 * ( gd_g_vec@gd_g_vec.T + self.eq_constraint[i, j] * Hess_g[i, j, :, :] ) # (d, d)
                self.x[i, j, :] = self.x[i, j, :] - np.linalg.solve( np.eye(self.x.shape[-1]) + (self.dt/self.epsilon) * Hess_gsqr, self.temp[i, j, :] ) 
                
        # self.error = np.linalg.norm(self.consensus - global_min, ord = 2)/np.sqrt(self.x.shape[-1])
        # self.error = np.linalg.norm(self.consensus - global_min, ord =2, axis = -1)/np.sqrt(self.x.shape[-1]) # (M, 1)
        # self.error = np.linalg.norm(self.x - np.tile(global_min, (1, self.x.shape[1], 1)), ord =2, axis = -1)/np.sqrt(self.x.shape[-1])
        self.error = np.linalg.norm(self.consensus - np.tile(global_min, (self.x.shape[0], 1, 1)), ord =2, axis = -1)/np.sqrt(self.x.shape[-1])