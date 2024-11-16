from mirrorcbx.utils import ExperimentConfig
from mirrorcbx.regularization import regularize_objective
from mirrorcbx.objectives import norm_sphere, data_fidelity
from cbx.scheduler import param_update
import numpy as np

def select_experiment(conf_path):
    return Linear_Experiment(conf_path)

#%%

class sparsity_scheduler(param_update):
    def __init__(self, target_sp = 1., minimum = 10, factor = 1.05, maximum=1e15):
        super().__init__(maximum=maximum, minimum=minimum)
        self.target_sp = target_sp
        self.factor = factor
        self.maximum = maximum
        
    def update(self, dyn):
        sp = np.linalg.norm(dyn.consensus, ord=1, axis=-1)
        #dyn.alpha *= np.where(sp < self.target_sp, self.factor, 1)
        dyn.alpha *= np.clip((0.6 + 1/(1 + np.exp(sp))), a_min=1., a_max=self.factor)
        self.ensure_max(dyn)

#%%
def shrink(z, l):
    return np.sign(z) * np.maximum(np.abs(z) - l, 0)
#%%
class Linear_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_reg()
        self.problem_d = self.d
        
    def set_reg(self,):
        dname = self.config.dyn.name
        if dname in ['MirrorCBO', 'KickingMirrorCBO', 'PolarMirrorCBO']:
            self.dyn_kwargs['mirrormap'] = {
                 'name':'ElasticNet',
                 'lamda':getattr(self.config.mirrormap, 'lamda', 1.)
                 }
        
    
    def get_objective(self,):
        self.center = np.zeros((self.d,))
        self.center[0] = 1.5
        self.p = 1.5
        
        if self.obj == 'norm_sphere':
            ob = norm_sphere(center = 
                               self.center[None,None,:], 
                               p = self.p)
        elif self.obj == 'data_fid':
            self.A = np.ones((1, self.problem_d))
            self.A[0,0] = 2
            self.f = 1
            ob = data_fidelity(self.f, self.A)
        else:
            raise ValueError('Unknown Objective')
        
        reg = getattr(self.config, 'reg', None)
        lamda = 0.1 if reg is None else getattr(reg, 'plamda', 0.0)
        if self.config.dyn.name in ['PenalizedCBO', 'MirrorCBO'] and lamda > 0:
            ob = regularize_objective(
                ob, 
                {'name':getattr(reg, 'name', 'L1')},
                lamda=lamda,
                lamda_broadcasted=True)
        if (reg is not None) and getattr(reg, 'dualize', False):
            def dist_to_box(x, lamda):
                z = np.clip(x, a_min=-1, a_max=1)
                return 0.5*lamda*np.linalg.norm(x-z, axis=-1)**2
            def ob(y):
                return - (self.f * y).sum(axis=-1) + dist_to_box(y@self.A, lamda=lamda)
            
            self.d = 1
        return ob
    
    # def get_scheduler(self,):
    #     sdyn = getattr(self.config, 'scheduler', {'name':''})
    #     skwargs = {k:v for k,v in sdyn.items() if k not in ['name']}
    #     # if sdyn['name'] in scheduler_dict:
    #     #     sched = scheduler_dict[sdyn.name](**skwargs)
    #     # else:
    #     #     sched = None
            
    #     if self.config.dyn.name in ['MirrorCBO']:
    #         sched = sparsity_scheduler(**skwargs)
            
           
    #     return sched
    
    def set_diffs(self, x, c, const_minimizer):
        c = self.dual_to_primal(c)
        super().set_diffs(x, c, const_minimizer)

    
    def get_minimizer(self,):
        reg = getattr(self.config, 'reg', None)
        d = self.d
        if (reg is not None) and getattr(reg, 'dualize', False):
            d += 1
        z = np.zeros(d)
        z[0] = 0.5
        return z
    
    def eval_success(self, c, const_minimizer):
        c = self.dual_to_primal(c)
        return super().eval_success(c, const_minimizer)
        
    def dual_to_primal(self, c):
        reg = getattr(self.config, 'reg', None)
        lamda = 0.1 if reg is None else getattr(reg, 'plamda', 0.0)
        if (reg is not None) and getattr(reg, 'dualize', False):
            c = lamda * shrink(c@self.A, 1)
        return c
    
    # def set_diffs(self, x, c, const_minimizer):
    #     for z, n in [(x, 'diff'), (c, 'diff_c')]:
    #         if z is not None:
    #             error = #
    #             setattr(
    #                 self, 
    #                 n, 
    #                 get_error_min(const_minimizer, self.R * z[..., :-1]).mean(axis=-1).squeeze()
    #             )

    # def eval_success(self, c, const_minimizer):
    #     num_scc = (self.diff_c[-1] < self.tol).sum()
    #     return {'num': num_scc, 'rate': num_scc/c.shape[1]}