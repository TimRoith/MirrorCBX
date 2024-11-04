from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.regularization import regularize_objective
from mirrorcbx.objectives import norm_sphere
import numpy as np

#%%
class NormSphere_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_reg()
        
    def set_reg(self,):
        dname = self.config.dyn.name
        if dname == 'MirrorCBO':
            self.dyn_kwargs['mirrormap'] = {
                 'name':'ElasticNet',
                 'lamda':getattr(self.config.reg, 'lamda', 1.)
                 }
        
    
    def get_objective(self,):
        self.center = np.zeros((self.d,))
        self.center[0] = 1.5
        self.p = 1.5
        
        if self.obj == 'norm_sphere':
            return norm_sphere(center = 
                               self.center[None,None,:], 
                               p = self.p)
        else:
            raise ValueError('Unknown Objective')
    
    def get_minimizer(self,):
        if self.obj == 'Ackley-A' and self.constr == 'Hyperplane-A':
            if self.d == 3:
                return  1/(self.d) * np.ones((self.d,))
        raise ValueError('The minimizer is not known for the current config')
        