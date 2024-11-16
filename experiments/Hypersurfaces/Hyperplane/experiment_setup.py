from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.mirrormaps import ProjectionHyperplane, MirrorMaptoPostProcessProx
from mirrorcbx.regularization import regularize_objective
from cbx.objectives import Ackley
import numpy as np

#%%
def select_experiment(conf_path):
    return Ackley_Experiment(conf_path)

#%%
class Ackley_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_constr()
        
    def set_constr(self,):
        self.constr = self.config.problem.constr
        if self.constr == 'Hyperplane-A':
            self.a = np.ones(self.d)
            self.b = 1
            dname = self.config.dyn.name
            if dname == 'MirrorCBO':
                self.dyn_kwargs['mirrormap'] = {
                    'name':'ProjectionHyperplane', 
                    'a': self.a,
                    'b': self.b
                    }
            elif dname == 'SphereCBO':
                self.dyn_kwargs['sdf'] = {
                    'name' : 'plane',
                    'a': self.a,
                    'b': self.b
                }
                
            elif dname == 'ProxCBO':
                pp = MirrorMaptoPostProcessProx(ProjectionHyperplane)(a=self.a,b= self.b)
                self.dyn_kwargs['post_process'] = pp

            elif dname == 'DriftConstrainedCBO':
                self.dyn_kwargs['constraints'] = [{
                    'name' : 'plane',
                    'a': self.a,
                    'b': self.b
                }]

            elif dname == 'RegCombinationCBO':
                self.dyn_kwargs['constraints'] = [{
                    'name' : 'plane',
                    'a': self.a,
                    'b': self.b
                }]
        else:
            raise ValueError('Unknown constraint: ' +str(self.constr))
    
    def get_objective(self,):        
        if self.obj == 'Ackley-A':
            v = 0.4 * np.ones((1,1, self.d))
            f = Ackley(minimum=v, c=np.pi*4, b=0.1)
        else:
            raise ValueError('Unknown objective ' + str(self.obj))
            
        if self.config.dyn.name == 'PenalizedCBO':
            reg = getattr(self.config, 'reg', None)
            lamda = 0.1 if reg is None else getattr(reg, 'lamda', 0.1)
            
            f = regularize_objective(
                f, 
                {'name':'Plane', 'a': self.a, 'b': self.b},
                lamda=lamda)
        return f
    
    def get_minimizer(self,):
        if self.obj == 'Ackley-A' and self.constr == 'Hyperplane-A':
            if self.d == 3:
                return  1/(self.d) * np.ones((self.d,))
        raise ValueError('The minimizer is not known for the current config')