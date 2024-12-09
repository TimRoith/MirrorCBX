from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.mirrormaps import ProjectionSphere, MirrorMaptoPostProcessProx
from mirrorcbx.regularization import regularize_objective
import cbx.utils.resampling as rsmp
from .phase_retrieval import operator, objective, get_error_min, WirtingerFlowBackTracking
from cbx.objectives import Ackley
from cbx.scheduler import multiply
import numpy as np
from omegaconf import OmegaConf

def select_experiment(conf_path):
    cfg = OmegaConf.load(conf_path)
    if (
        hasattr(cfg, 'problem') and 
        hasattr(cfg.problem, 'name') and 
        cfg.problem.name == 'phase'
        ):
        return PhaseRetrieval_Experiment(conf_path)
    return Ackley_Experiment(conf_path)

#%%
class PhaseRetrieval_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_constr()
        
    def set_problem_kwargs(self,):
        self.obj = self.config.problem.obj
        self.d   = self.config.problem.d + 1
        self.tol = getattr(self.config.problem, 'tol', 0.1)
        
    def set_dyn_kwargs(self,):
        cdyn = self.config['dyn']
        if cdyn.name == 'Wirtinger':
            self.dyn_cls = WirtingerFlowBackTracking
            self.dyn_kwargs = {k:v for k,v in cdyn.items() if k not in ['name']}
            self.Wirtinger = True
        else:
            self.Wirtinger = False
            super().set_dyn_kwargs()
         
    # def eval_Wirtinger_Flow(self,):
    #     #%% compute Wirtinger Flow
    #     x = WirtingerFlow(
    #         self.f, self.y, 
    #         max_it=10000, 
    #         tau_0 = 10, mu_max=1.
    #     )
    #     e = get_error_min(self.get_minimizer(), x)
    #     success = e < self.config.success.tol
    #     return {'success': success}
        
        
    def set_constr(self,):
        self.constr = self.config.problem.constr
        if self.constr == 'Sphere':
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
                pp = MirrorMaptoPostProcessProx(ProjectionSphere)()
                self.dyn_kwargs['post_process'] = pp

            elif dname == 'DriftConstrainedCBO':
                self.dyn_kwargs['constraints'] = [{
                    'name' : 'sphere',
                }]
            elif dname == 'RegCombinationCBO':
                self.dyn_kwargs['constraints'] = [{
                    'name' : 'sphere',
                }]

        else:
            raise ValueError('Unknown constraint: ' +str(self.constr))

    def get_objective(self,):
        prb = self.config.problem
        d, M, sigma_noise = (prb.d, prb.M, prb.sigma_noise)

        # get frame vectors
        f = np.random.normal(0, 1, (M, d))
        f = f/np.linalg.norm(f, axis=-1, keepdims=True)
        self.f = f
        self.x_true = np.random.normal(0, 1, (d,))
        op = operator(f)
        y = op(self.x_true) + sigma_noise * np.random.normal(0, 1, size=(M,))
        self.y = y
        if self.config.dyn.name == 'Wirtinger':
            self.dyn_kwargs['y'] = y
            self.dyn_kwargs['f'] = f
        
        ob = objective(y, f)
        self.R = ob.R
        return ob
    
    def get_scheduler(self,):
        return multiply(factor=1.05, maximum=1e18)
    
    def get_minimizer(self,):
        return self.x_true
    
    def set_diffs(self, x, c, x_true):
        for z, n in [(x, 'diff'), (c, 'diff_c')]:
            if z is not None:
                if not self.Wirtinger:
                    dd = get_error_min(x_true, self.R * z[..., :-1]).mean(axis=-1).squeeze()
                else:
                    dd = get_error_min(x_true, z).mean(axis=-1).squeeze()
                    
                if not hasattr(self, n): 
                    setattr(self, n, dd/self.num_runs)
                else:
                    setattr(self, n, 
                            ((self.num_runs - self.M) * getattr(self, n) + dd)/
                            self.num_runs
                            )
    def eval_success(self, c, x_true):
        if not self.Wirtinger:
            diff = get_error_min(x_true, self.R * c[..., :-1], p=np.inf)
            M = c.shape[0]
        else:
            diff = get_error_min(x_true, c[None, None, :], p=np.inf)
            M = 1

            
        idx = np.where(diff < self.tol)[0]
        
        return {'num': len(idx), 
                'rate': len(idx)/M, 
                'normdiff': diff,
                'idx':idx}
        
    # def eval_run(self, dyn):
    #     x_true = self.get_minimizer()
    #     c = np.array(dyn.history['consensus'])
        
        
    #     success = e[-1] < self.config.success.tol
    #     return {'consensus_diff': e, 'success': success}

class Ackley_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_constr()
        
    def set_constr(self,):
        self.constr = self.config.problem.constr
        if self.constr == 'Sphere':
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
                pp = MirrorMaptoPostProcessProx(ProjectionSphere)()
                self.dyn_kwargs['post_process'] = pp
            elif dname == 'DriftConstrainedCBO':
                self.dyn_kwargs['constraints'] = [{
                    'name' : 'sphere',
                }]
            elif dname == 'RegCombinationCBO':
                self.dyn_kwargs['constraints'] = [{
                    'name' : 'sphere',
                }]
            
        else:
            raise ValueError('Unknown constraint: ' +str(self.constr))
    
    def get_objective(self,):
        self.set_resampling()
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
                {'name':'Sphere',},
                lamda=lamda)
        return f
    
    def set_resampling(self,):
        if self.config.dyn.name in ['DriftConstrainedCBO', 'RegCombinationCBO']:
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

            self.dyn_kwargs['post_process'] = lambda dyn: self.resampling(dyn)
    
    def get_minimizer(self,):
        if self.obj == 'Ackley-B' and self.constr == 'Sphere':
            if self.d == 3 or self.d == 20:
                return  1/(self.d**0.5) * np.ones((self.d,))
        elif self.obj == 'Ackley-C':
            v = np.zeros((1,1,self.d))#
            v[0,0,-1] = 1
            return v
        raise ValueError('The minimizer is not known for the current config')