import numpy as np
from scipy.special import logsumexp

from .particledynamic import ParticleDynamic
import mirrorcbo as mcbo

#%% CBO
class MirrorCBO2(ParticleDynamic):
    def __init__(self,x, V, noise,\
                 beta = 1.0, noise_decay=0.0, diff_exp=1.0,\
                 tau=0.1, sigma=1.0, lamda=1.0, MirrorFct=None ,M=None,\
                 overshoot_correction=False, heavi_correction=False):
        
        super(MirrorCBO2, self).__init__(x, V, beta = beta)
        
        # additional parameters
        self.noise_decay = noise_decay
        self.tau = tau
        self.beta = beta
        self.diff_exp = diff_exp
        self.noise = noise
        self.sigma = sigma
        self.lamda = lamda
        self.overshoot_correction = overshoot_correction
        self.heavi_correction = heavi_correction

        self.M = M
        # what is the difference between M and num_particles? [TO CHECK]
        if self.M is None:
            self.M = self.num_particles

        self.q = self.num_particles// self.M
        
        self.MirrorFct = MirrorFct 
        if self.MirrorFct is None:
            self.MirrorFct = mcbo.functional.L2()
        
        # compute mean for init particles
        self.m_beta = self.compute_mean()
        self.update_diff = float('inf')
        self.m_diff = self.x - self.m_beta
        self.y = self.MirrorFct.grad(self.x)
    
    def step(self,time=0.0):
        ind = np.random.permutation(self.num_particles)
        
        for i in range(self.q):
            loc_ind = ind[i*self.M:(i+1)*self.M]
            self.m_beta = self.compute_mean(loc_ind)
                        
            x_old = self.x.copy()
            self.m_diff = self.x - self.m_beta
            
            #update step
            self.y[loc_ind,:] = self.y[loc_ind,:] -\
                                self.lamda * self.tau * self.m_diff[loc_ind, :] +\
                                np.sqrt(self.MirrorFct.hessian(self.x[loc_ind, :])) *\
                                self.sigma * self.noise(self.m_diff[loc_ind, :])
            
            # compare with np.linalg.inv(self.MirrorFct.grad_Hessian(self.y[loc_ind, :]))
            # if phi (mirrorfct) is smooth, 
            # np.linalg.inv(self.MirrorFct.grad_Hessian(self.y[loc_ind, :])) = self.MirrorFct.hessian(self.x[loc_ind, :])
                                
            self.x[loc_ind,:] = self.MirrorFct.grad_conj(self.y[loc_ind,:])                                
            self.update_diff = np.linalg.norm(self.x - x_old)
        
        
    def compute_mean(self, ind=None):
        if ind is None:
            ind = np.arange(self.num_particles)
        
        m_beta = np.zeros(self.x.shape)
        # update energy
        self.energy = self.V(self.x)[ind]
        # V_min = np.min(self.energy)
        
        # for j in ind:
        weights = - self.beta * self.energy
        coeffs = np.expand_dims(np.exp(weights - logsumexp(weights)), axis=1)
        m_beta[ind,:] = np.sum(self.x[ind,:]*coeffs,axis=0)
        
        return m_beta