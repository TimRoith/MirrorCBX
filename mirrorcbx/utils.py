from cbx.dynamics import CBO
from mirrorcbx.dynamics import (
    MirrorCBO, SphereCBO, DriftConstrainedCBO, 
    KickingMirrorCBO, PolarMirrorCBO, 
    RegCombinationCBO
)
from mirrorcbx.regularization import regularization_paramter_scheduler
from omegaconf import OmegaConf
import numpy as np
from collections import OrderedDict
from functools import reduce
from cbx.scheduler import multiply, scheduler, effective_sample_size
import cbx.utils.success as scc
#%%


param_group_dict = {
    'dyn': [
        'dt',
        'alpha',
        'lamda',
        'noise',
        'sigma',
        'M',
        'N',
        'd',
        'max_it',
        'batch_args',
    ],
    'scheduler': ['all'],
    'init': ['all'],
    'problem': ['all'],
}
param_groups = ['dyn', 'scheduler', 'init', 'problem']

def conf_to_dict(conf):
    params = OrderedDict()
    for pg in param_groups:
        group = getattr(conf, pg, None)
        for k in param_group_dict[pg]:
            if group and hasattr(group, k):
                params[k] = group[k]
            elif group and k == 'all':
                params[pg] = str(group)
            elif k not in ['all']:
                params[k] = None
    return params

def save_param_dict_to_table(param_dict, file_name):
    with open(file_name,'w') as file:
        for l in [param_dict.keys(), param_dict.values()]:
            file.write(
                reduce(lambda a,b: str(a) + ' & ' + str(b), l)
            )
            file.write(r'\\')
            file.write('\n')
            
def save_conf_to_table(conf):
    pdict = conf_to_dict(conf.config)
    save_param_dict_to_table(pdict, conf.path + 'results/' + conf.name() + '_params.txt')
    
def init_uniform(low=0, high=1., size=(1,1,1)):
    return np.random.uniform(low, high, size)   

def init_normal(mean=0, std=1., size=(1,1,1)):
    return np.random.normal(mean, std, size)

def init_sphere(mean=0, std=1., size=(1,1,1)):
    z = np.random.normal(mean, std, size)
    return z/np.linalg.norm(z,axis=-1, keepdims=True)

def init_sphere_half(mean=0, std=1., size=(1,1,1)):
    z = np.random.normal(mean, std, size)
    z[..., -1] *= np.sign(z[..., -1])
    return z/np.linalg.norm(z,axis=-1, keepdims=True)

def init_sparse_uniform(low=0, high=1., p = 1, size=(1,1,1)):
    z = init_uniform(low=low, high=high, size=size)
    c = np.random.choice(a=[False, True], size=size, p=[p, 1-p])
    z[c] = 0.
    return z

def init_simplex_uniform(size=(1,1,1)):
    z = np.random.exponential(size=size)
    return z/z.sum(axis=-1, keepdims=True)

def init_simplex(size):
    z = np.ones(size)/size[-1]
    z += 0.15 * (1/size[-1]) * np.random.uniform(-1,1, size=size)
    return projection_simplex(z)

init_dict = {
    'uniform': init_uniform,
    'normal': init_normal, 
    'sphere':init_sphere, 
    'sphere-half': init_sphere_half,
    'sparse-uniform': init_sparse_uniform,
    'simplex': init_simplex_uniform}

dyn_dict = {'MirrorCBO':MirrorCBO, 'SphereCBO':SphereCBO, 
            'ProxCBO': CBO, 'PenalizedCBO': CBO, 'CBO':CBO,
            'DriftConstrainedCBO': DriftConstrainedCBO,
            'KickingMirrorCBO': KickingMirrorCBO,
            'PolarMirrorCBO': PolarMirrorCBO,
            'RegCombinationCBO': RegCombinationCBO}

scheduler_dict = {'multiply': multiply, 'effective': effective_sample_size}


def run_down_dict(d):
    l = []
    p = []
    if hasattr(d, 'keys'):
        for k in d.keys():
            ll, pp = run_down_dict(d[k])
            l += ll
            p += [k + '.' * (len(ppp)>0) + ppp for ppp in pp]
    else:
        return [d], ['']
    return l, p

class ExperimentConfig:
    def __init__(self, config_path, name_ext = ''):
        self.config = OmegaConf.load(config_path)
        self.cp = config_path
        self.name_ext = name_ext
        self.path = config_path[:config_path.find('params/')]
        self.set_dyn_kwargs()
        self.set_problem_kwargs()
        self.set_sweeps()
        
    def name(self,):
        return self.cp[
            self.cp.find('params/') + len('params/'):
        ].split('.')[0] + self.name_ext
        
    def get_objective(self,):
        raise NotImplementedError('This class does not implement the function: ' + 
                                  'get_objective')
    
    def set_dyn_kwargs(self,):
        cdyn = self.config['dyn']
        self.dyn_cls = dyn_dict[cdyn['name']]
        self.dyn_kwargs = {k:v for k,v in cdyn.items() if k not in ['name']}
        
    def set_sweeps(self,):
        self.sweeps = getattr(self.config, 'sweeps', None)
        if self.sweeps is not None:
            self.sweep_list = run_down_dict(self.sweeps)
            self.num_sweeps = len(self.sweep_list[0][0])
        else:
            self.sweep_list = None
            self.num_sweeps = 1
            
    
    def set_sweep(self):
        if not hasattr(self, 'sweep_ctr'): self.sweep_ctr = -1
        self.sweep_ctr += 1
        if self.sweeps is not None:
            self.name_ext = str(self.sweep_ctr)
            print(20 * '<>')
            print('Starting sweep')
            for i, n in enumerate(self.sweep_list[1]):
                val = self.sweep_list[0][i][self.sweep_ctr]
                setattr(self, n, val)
                print('Setting ' + n + ' to ' + str(val))
            print(20 * '<>')
        
        # save config
        save_conf_to_table(self)
        
        
        
    def set_problem_kwargs(self,):
        self.obj = self.config.problem.obj
        self.d   = self.config.problem.d
        self.tol = getattr(self.config.problem, 'tol', 0.1)
        
    def get_scheduler(self,):
        sdyn = getattr(self.config, 'scheduler', {'name':''})
        skwargs = {k:v for k,v in sdyn.items() if k not in ['name']}
        if sdyn['name'] in scheduler_dict:
            sched = scheduler_dict[sdyn.name](**skwargs)
        else:
            sched = None
            
        if self.config['dyn']['name'] == 'PenalizedCBO': 
            reg_sched_kwargs = getattr(self.config, 'reg_sched', {})
            if reg_sched_kwargs.get('use', True):
                reg_sched = regularization_paramter_scheduler(**reg_sched_kwargs)
                sched = scheduler([sched, reg_sched])
           
        return sched

    def evaluate_dynamic(self, dyn):
        # update number of runs
        if not hasattr(self, 'num_runs'): self.num_runs = 0
        self.M = dyn.M
        self.num_runs += dyn.M

        const_minimizer = self.get_minimizer()
        fname = self.path + 'results/' + self.name()
        x = np.array(dyn.history['x'])[1:, ...] if 'x' in dyn.history else None
        c = np.array(dyn.history['consensus'])[1:, ...] if 'consensus' in dyn.history else None

        self.set_diffs(x, c, const_minimizer)
        for n in [('diff'), ('diff_c')]:
            if hasattr(self, n):
                np.savetxt(fname + '_' + n + '.txt', getattr(self, n))
        
        # evaluate success
        scc_eval = self.eval_success(dyn.consensus, const_minimizer)
        if hasattr(self, 'scc_eval'):
            self.scc_eval['num'] += scc_eval['num']
            self.scc_eval['rate'] = self.scc_eval['num']/self.num_runs
        else:
            self.scc_eval = scc_eval

        np.savetxt(fname + '_scc.txt', np.array([self.scc_eval['rate'], self.tol]))
        print('Success rate: ' + str(self.scc_eval['rate'] ), flush=True)
        self.set_energy(dyn)
        
    def set_diffs(self, x, c, const_minimizer):
        for z, n in [(x, 'diff'), (c, 'diff_c')]:
            if z is not None:
                dd = np.linalg.norm(z - const_minimizer, axis=-1).sum(axis=(-2,-1))
                if not hasattr(self, n): 
                    setattr(self, n, dd/self.num_runs)
                else:
                    setattr(self, n, 
                            ((self.num_runs - self.M) * getattr(self, n) + dd)/
                            self.num_runs
                            )

    def eval_success(self, c, const_minimizer):
        scc_eval = scc.dist_to_min_success(c, const_minimizer, tol = self.tol)
        return scc_eval
    
    def set_energy(self, dyn):
        e = dyn.history.get('energy', None)
        if e is not None:
            e = np.array(e)
            e = np.sum(e, axis=(-1,-2))
            if not hasattr(self, 'energy'):
                self.energy = e/self.num_runs
            else:
                self.energy = ((self.num_runs - self.M) * self.energy + e)/self.num_runs
            
    def init_x(self,):
        dyn = self.config.dyn
        name = self.config.init.name
        kwargs = {k:v for k,v in self.config.init.items() if k not in ['name']}
        kwargs['size'] = (dyn.M, dyn.N, self.d)
        return init_dict[name](**kwargs)
    
    def reset_eval(self,):
        for n in ('diff', 'diff_c', 'energy', 'scc_eval'):
            if hasattr(self, n):
                delattr(self, n)
    

def projection_simplex(v):
    size = v.shape
    v = v.reshape((-1, size[-1])).T
    v = projection_simplex_sort_2d(v)
    return np.reshape(v.T, shape=size)
    
def projection_simplex_sort_2d(v, z=1):
    """v array of shape (n_features, n_samples)."""
    p, n = v.shape
    u = np.sort(v, axis=0)[::-1, ...]
    pi = np.cumsum(u, axis=0) - z
    ind = (np.arange(p) + 1).reshape(-1, 1)
    mask = (u - pi / ind) > 0
    rho = p - 1 - np.argmax(mask[::-1, ...], axis=0)
    theta = pi[tuple([rho, np.arange(n)])] / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w
        

#%%
class consensus_stagnation_pp:
    def __init__(self, thresh=1e-4, loss_thresh=0.1, 
                 indep_sigma=0.01, patience = 20,
                 reset_alpha=1e18, var_name='y',
                 update_thresh=1e-3,
                 decrease_sigma=.99):
        self.thresh = thresh
        self.loss_thresh = loss_thresh
        self.indep_sigma = indep_sigma
        self.best_energy = None
        self.patience = patience
        self.reset_alpha = reset_alpha
        self.energies = []
        self.min_decrease = 0.99
        self.var_name = var_name
        self.update_thresh = update_thresh
        self.decrease_sigma = decrease_sigma
        
    def __call__(self, dyn):
        wt = self.check_consensus_update(dyn)
        idx   = np.where(wt)
        var = getattr(dyn, self.var_name)
        if len(idx[0]) > 0:
            z = np.random.normal(0, 1, size = (len(idx[0]), dyn.N) + dyn.d)
            var[idx[0], ...] += self.indep_sigma * (dyn.dt**0.5) * z
            dyn.alpha[idx[0],...] = np.minimum(dyn.alpha[idx[0],...],
                                               self.reset_alpha
                                               )
            self.indep_sigma *= self.decrease_sigma
        setattr(dyn, self.var_name, np.clip(var, a_min=-100, a_max=100))
    
    def check_consensus_update(self, dyn):
        if not hasattr(self, 'consensus_updates'): self.consensus_updates = []
        wt = np.zeros((dyn.M,), dtype=bool)
        if hasattr(self, 'consensus_old'):
            self.consensus_updates.append(
                np.linalg.norm(dyn.consensus - self.consensus_old)
            )
            if len(self.consensus_updates) == self.patience:
                wt = np.array([self.consensus_updates]).mean(axis=-1) < self.update_thresh
                self.consensus_updates = []
        self.consensus_old = dyn.copy(dyn.consensus)
        return wt