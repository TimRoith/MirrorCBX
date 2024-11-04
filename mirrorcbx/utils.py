from cbx.dynamics import CBO
from mirrorcbx.dynamics import MirrorCBO, SphereCBO, DriftConstrainedCBO
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
    save_param_dict_to_table(pdict, conf.path + 'results/' + conf.config.name + '_params.txt')
    
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

init_dict = {
    'uniform': init_uniform,
    'normal': init_normal, 
    'sphere':init_sphere, 
    'sphere-half': init_sphere_half}
dyn_dict = {'MirrorCBO':MirrorCBO, 'SphereCBO':SphereCBO, 
            'ProxCBO': CBO, 'PenalizedCBO': CBO, 
            'DriftConstrainedCBO': DriftConstrainedCBO,}
scheduler_dict = {'multiply': multiply, 'effective': effective_sample_size}


class ExperimentConfig:
    def __init__(self, config_path):
        self.config = OmegaConf.load(config_path)
        self.path = config_path[:config_path.find('params/')]
        self.set_dyn_kwargs()
        self.set_problem_kwargs()
        
    def get_objective(self,):
        raise NotImplementedError('This class does not implement the function: ' + 
                                  'get_objective')
    
    def set_dyn_kwargs(self,):
        cdyn = self.config['dyn']
        self.dyn_cls = dyn_dict[cdyn['name']]
        self.dyn_kwargs = {k:v for k,v in cdyn.items() if k not in ['name']}
        
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
            reg_sched = regularization_paramter_scheduler(**reg_sched_kwargs)
            sched = scheduler([sched, reg_sched])
           
        return sched

    def evaluate_dynamic(self, dyn):
        # update number of runs
        if not hasattr(self, 'num_runs'): self.num_runs = 0
        self.num_runs += dyn.M

        const_minimizer = self.get_minimizer()
        fname = self.path + 'results/' + self.config.name
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
        print('Success rate: ' + str(self.scc_eval['rate'] ))

    def set_diffs(self, x, c, const_minimizer):
        for z, n in [(x, 'diff'), (c, 'diff_c')]:
            if z is not None:
                dd = np.linalg.norm(z - const_minimizer, axis=-1).mean(axis=(-2,-1))
                if not hasattr(self, n): 
                    setattr(self, n, dd)
                else:
                    setattr(self, n, 0.5 * (getattr(self, n) + dd)) # assumes equal sub run length

    def eval_success(self, c, const_minimizer):
        scc_eval = scc.dist_to_min_success(c, const_minimizer, tol = self.tol)
        return scc_eval
            
        
        
    def init_x(self,):
        dyn = self.config.dyn
        name = self.config.init.name
        kwargs = {k:v for k,v in self.config.init.items() if k not in ['name']}
        kwargs['size'] = (dyn.M, dyn.N, self.d)
        return init_dict[name](**kwargs)
        