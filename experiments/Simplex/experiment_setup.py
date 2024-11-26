from mirrorcbx.utils import ExperimentConfig, init_simplex_uniform, init_simplex
import numpy as np
import cbx
from cbx.scheduler import scheduler
from numpy.fft import fft, ifft, fftshift, ifftshift
from mirrorcbx.objectives import data_fidelity_L1
from mirrorcbx.utils import consensus_stagnation_pp
#%%
def select_experiment(conf_path):
    return Regression_Experiment(conf_path)
        
        
#%%
class Regression_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_reg()
        
    def set_reg(self,):
        dname = self.config.dyn.name
        if dname == 'MirrorCBO':
            self.dyn_kwargs['mirrormap'] = {
                 'name':'EntropySimplex',
                 }
    def set_problem_kwargs(self,):
        super().set_problem_kwargs()
        self.m         = self.config.problem.m
        self.noise_lvl = self.config.problem.noise_lvl
        
    def get_objective(self,):
        self.set_post_process()
        self.A = np.random.normal(0, 1, size = (self.m, self.d))
        self.x_true = init_simplex_uniform(size=(1, self.d))[0,:]
        self.b = self.A@self.x_true
        self.b += self.noise_lvl * np.random.normal(0,1, size=self.b.size)
        return data_fidelity_L1(self.b, self.A)
    
    def get_minimizer(self,):
        return self.x_true[None, None, :]
    
    def evaluate_dynamic(self, dyn):
        super().evaluate_dynamic(dyn)
        #self.set_energy(dyn)
        
    def set_post_process(self,):
        pp = getattr(self.config, 'postprocess', {'name':'default'})
        if pp['name'] == 'consensus_stagnation':
            self.dyn_kwargs['post_process'] = consensus_stagnation_pp(
                **{k:v for k,v in pp.items() if k not in ['name']}
            )
    
    # def set_energy(self, dyn):
    #     e = dyn.history.get('energy', None)
    #     if e is not None:
    #         e = np.array(e)
    #         e = np.mean(e, axis=(-1,-2))
    #         if not hasattr(self, 'energy'):
    #             self.energy = e
    #         else:
    #             self.energy = (self.energy + (self.num_runs-1) * e)/self.num_runs
            
    def set_diffs(self, x, c, const_minimizer):
        for z, n in [(x, 'diff'), (c, 'diff_c')]:
            if z is not None:
                dd = np.linalg.norm(z - const_minimizer, axis=-1, ord=1).mean(axis=(-2,-1))
                if not hasattr(self, n): 
                    setattr(self, n, dd)
                else:
                    setattr(self, n, 
                            (getattr(self, n) + (self.num_runs-1) * dd)/
                            self.num_runs
                            )