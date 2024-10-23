from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.mirrormaps import ProjectionHyperplane, MirrorMaptoPostProcessProx
from mirrorcbx.regularization import regularize_objective
from cbx.objectives import Ackley
import numpy as np

#%%
class Ackley_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_constr()
        
    def set_constr(self,):
        self.constr = self.config.problem.constr
        if self.constr == 'Sphere':
            self.a = np.ones(self.d)
            self.b = 1
            dname = self.config.dyn.name
            if dname == 'MirrorCBO':
                self.dyn_kwargs['mirrormap'] = {
                    'name':'ProjectionSphere',
                    }
            elif dname == 'SphereCBO':
                self.dyn_kwargs['sdf'] = {
                    'name' : 'sphere',
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
        else:
            raise ValueError('Unknown constraint: ' +str(self.constr))
    
    def get_objective(self,):        
        if self.obj == 'Ackley-B':
            v = 0.4 * np.ones((1,1, self.d))
            f = Ackley(minimum=v, c=np.pi*2, b=0.1)
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
        if self.obj == 'Ackley-B' and self.constr == 'Sphere':
            if self.d == 3 or self.d == 20:
                return  1/(self.d**0.5) * np.ones((self.d,))
        raise ValueError('The minimizer is not known for the current config')