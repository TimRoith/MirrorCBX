from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.mirrormaps import ProjectionSphere, MirrorMaptoPostProcessProx
from mirrorcbx.regularization import regularize_objective
from cbx.objectives import Ackley
from cbx.scheduler import multiply
import numpy as np
from quadproj import quadrics
from quadproj.project import project

#%%

class QuadricMirror:
    def __init__(self, A, b, c):
        self.Q = quadrics.Quadric(A, b, c)
        
    def grad(self, x):
        return x
    
    def grad_conj(self, y):
        y = np.clip(y, a_min=-100, a_max=100)
        x = project(self.Q, y)
        idx_c = np.where(~self.Q.is_feasible(x))
        if len(idx_c[0]) > 0:
           x[idx_c] = 1#project(self.Q, x[idx_c] * self.off)
        return x
    

#%%
class Ackley_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.A = np.eye(self.d)
        self.A[-1, -1] = 0
        self.b = np.zeros(self.d)
        self.b[-1] = -1
        self.c = 0
        self.set_constr()

        
        
    def set_constr(self,):
        self.constr = self.config.problem.constr
        dname = self.config.dyn.name
        if dname == 'MirrorCBO':           
            QM = QuadricMirror(self.A, self.b, self.c)
            self.dyn_kwargs['mirrormap'] = QM
        elif dname == 'ProxCBO':
            pp = MirrorMaptoPostProcessProx(QM)(self.A, self.b, self.c)
            self.dyn_kwargs['post_process'] = pp
        elif dname == 'DriftConstrainedCBO':
            self.dyn_kwargs['constraints'] = [{
                'name' : 'quadric', 'A': self.A, 'b': self.b, 'c': self.c,
            }]

    def get_objective(self,):
        if self.obj == 'Ackley-C':
            v = np.zeros((1,1,self.d))#
            v[0,0,-1] = 1
            f = Ackley(minimum=v, c=3 * 2 * np.pi, b=0.2*3)
            const_minimizer = v
        elif self.obj == 'Ackley-B':
            v = 0.4 * np.ones((1,1, self.d))
            f = Ackley(minimum=v, c=np.pi*2, b=0.1)
        else:
            raise ValueError('Unknown objective ' + str(self.obj))
            
        if self.config.dyn.name == 'PenalizedCBO':
            reg = getattr(self.config, 'reg', None)
            lamda = 0.1 if reg is None else getattr(reg, 'lamda', 0.1)
            
            f = regularize_objective(
                f, 
                {'name':'Quadric', 'A': self.A, 'b': self.b, 'c': self.c},
                lamda=lamda)
        return f
    
    
    def get_minimizer(self,):
        if self.d == 20:    
            x = np.zeros(self.d)
            x[:-1] = 0.3542
            x[-1]  = 2.3839
            return x
        else:
            raise ValueError('Constrained Minimizer not known for d=' + 
                             str(self.d))