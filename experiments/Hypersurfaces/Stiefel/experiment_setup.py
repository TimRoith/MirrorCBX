from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.mirrormaps import ProjectionSphere, MirrorMaptoPostProcessProx, ProjectionStiefel
from mirrorcbx.regularization import regularize_objective
from cbx.objectives import Ackley
from cbx.scheduler import multiply
import numpy as np
import cbx.utils.resampling as rsmp

def select_experiment(cfg):
    return Ackley_Experiment(cfg)

#%%
class Ackley_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_constr()
        
    def set_problem_kwargs(self,):
        super().set_problem_kwargs()
        self.M = self.config.dyn.M
        self.k = self.config.problem.k
        
    def set_random_ortho_matrices(self,):
        X = np.random.uniform(-1,1, size=(self.M, self.k, self.d))
        U, _ , Vh = np.linalg.svd(X, full_matrices=False)
        self.RO = (U@Vh).reshape(self.M, 1, self.k * self.d)
        

    def set_constr(self,):
        self.set_resampling()
        self.constr = self.config.problem.constr
        dname = self.config.dyn.name
        if dname == 'MirrorCBO':           
            QM = ProjectionStiefel((self.k, self.d))
            self.dyn_kwargs['mirrormap'] = QM
        elif dname == 'ProxCBO':
            pp = MirrorMaptoPostProcessProx(ProjectionStiefel)((self.k, self.d))
            self.dyn_kwargs['post_process'] = pp
        elif dname == 'SphereCBO':
            self.dyn_kwargs['sdf'] = {
                'name' : 'Stiefel', 'nk': (self.k,self.d)
            }
            
    def init_x(self,):
        if self.k < self.d:
            raise ValueError('k cannot be smaller than d for uniform Stiefel init')
        
        dyn = self.config.dyn
        z = np.random.normal(0,1,size=(dyn.M* dyn.N, self.k, self.d))
        z = z@(np.linalg.cholesky(np.linalg.inv(np.moveaxis(z, -1,-2)@z)))
        return z.reshape(dyn.M, dyn.N, self.k*self.d)
            

    def get_objective(self,):
        self.set_random_ortho_matrices()
        if self.obj == 'Ackley-B':
            f = Ackley(minimum=self.RO, c=np.pi*2, b=0.1)
        else:
            raise ValueError('Unknown objective ' + str(self.obj))
            
        # if self.config.dyn.name == 'PenalizedCBO':
        #     reg = getattr(self.config, 'reg', None)
        #     lamda = 0.1 if reg is None else getattr(reg, 'lamda', 0.1)
            
        #     f = regularize_objective(
        #         f, 
        #         {'name':'Quadric', 'A': self.A, 'b': self.b, 'c': self.c},
        #         lamda=lamda)
        # self.set_resampling()
        return f
    
    def eval_success(self, x, x_true):
        norm_diff = np.linalg.norm((x_true - x), 
                                   axis=-1, ord = np.inf)
        idx = np.where(norm_diff < self.tol)[0]
        
        return {'num': len(idx), 
                'rate': len(idx)/x.shape[0], 
                'normdiff':norm_diff,
                'idx':idx}
    
    def set_resampling(self,):
        if self.config.dyn.name in ['DriftConstrainedCBO', 'RegCombinationCBO', 
                                    'MirrorCBO', 'ProxCBO', 'PenalizedCBO']:
            ckwargs = {'patience': 20, 'update_thresh':0.01}
            rkwargs = {'sigma_indep':0.3, 'var_name':'x', 
                       'track_best_consensus':False}
            if hasattr(self.config, 'resampling'):
                cr = self.config.resampling
                for kwargs in [ckwargs, rkwargs]:
                    for k in kwargs.keys():
                        if hasattr(cr, k):
                            kwargs[k] = getattr(cr, k)
            
            self.resampling =  rsmp.resampling(
                [rsmp.consensus_stagnation(**ckwargs,)],
                **rkwargs,
                )
            
            if self.config.dyn.name == 'ProxCBO':
                prox = MirrorMaptoPostProcessProx(ProjectionStiefel)((self.k,self.d))
                def pp_comb(dyn):
                    prox(dyn)
                    self.resampling(dyn)
                    
                self.dyn_kwargs['post_process'] = pp_comb
            else:
                self.dyn_kwargs['post_process'] = lambda dyn: self.resampling(dyn)
    

    
    def get_minimizer(self,):
        return self.RO