from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.regularization import regularize_objective
from mirrorcbx.objectives import norm_sphere, data_fidelity
from cbx.objectives import Rastrigin
import numpy as np

def select_experiment(conf_path):
    return Rastrigin_Experiment(conf_path)

#%%
class Rastrigin_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_reg()
        self.problem_d = self.d
        
    def set_reg(self,):
        dname = self.config.dyn.name
        if dname in ['MirrorCBO']:
            self.dyn_kwargs['mirrormap'] = {
                 'name':'L2',
                 'lamda':getattr(self.config.mirrormap, 'lamda', 1.)
                 }
        
    def get_objective(self,):
        ob = Rastrigin()
        return ob

    
    def get_minimizer(self,):
        z = np.zeros(self.d,)
        return z