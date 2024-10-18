from .pdyn import CBXDynamic
import numpy as np

#%% CBO
class ProjectionCBO(CBXDynamic):
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
        self.drift = self.x[self.particle_idx] - self.consensus # (M, N, d) / note that consensus \in (M, 1, d)
        # print(self.drift.shape)

        # define projection maps
        PV = np.zeros(self.x.shape)
        PB = np.zeros(self.x.shape)
        diffv = np.zeros(self.x.shape)

        for i in range(self.M): # add this after checking this works for M = 1
            for j in range(self.N):
                w = self.x[i, j, :] # (i, j)th particle in d-dim
                p = np.eye(self.x.shape[-1]) - np.outer(w, w) / np.linalg.norm(w, 2) # (d, d)
                PV[i, j, :] = p @ self.consensus[i, 0, :] # (d, 1)
                PB[i, j, :] = p @ np.random.randn(self.x.shape[-1]) * np.sqrt(self.dt)
                diffv[i, j, :] = np.outer(self.drift[i, j, :], self.drift[i, j, :]) @ w
        
        self.x[self.particle_idx] = ( self.x[self.particle_idx] +
                self.dt * self.lamda * PV + np.sqrt(self.dt) * self.sigma * np.multiply(self.drift, PB) -
                self.dt * self.sigma**2 / 2 * diffv )
        denom = np.sqrt( np.sum(self.x[self.particle_idx]**2, axis = -1) )[:, :, None]
        self.x[self.particle_idx] = np.divide( self.x[self.particle_idx], 
                                    denom, out = np.zeros_like(self.x[self.particle_idx]), where=denom != 0 )
        
        self.error = np.linalg.norm(self.consensus - np.tile(global_min, (self.x.shape[0], 1, 1)), ord =2, axis = -1)/np.sqrt(self.x.shape[-1])
        # self.error = np.linalg.norm(self.x - np.tile(global_min, (1, self.x.shape[1], 1)), ord =2, axis = -1)/np.sqrt(self.x.shape[-1])
        