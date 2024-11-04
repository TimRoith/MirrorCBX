import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
#%%
problem = 'ElasticNet:deconvolution'
experiment_name = 'Deconvolution_Experiment'#'Ackley_Experiment' 
params  = 'mirror_params'

#%%
CFG = getattr(
    importlib.import_module(
        problem.replace(':','.') + '.experiment_setup'
    ),
    experiment_name
)

param_path = problem.replace(':','/') + '/params/' + params + '.yaml'
conf = CFG(param_path)
save_conf_to_table(conf)
#%%
for rep in range(getattr(conf.config, 'reps', 1)):
    f = conf.get_objective()
    x = conf.init_x()
    
    dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
    sched = conf.get_scheduler()
    dyn.optimize(sched=sched)
    #%% Evaluate experiment
    conf.evaluate_dynamic(dyn)

#%% plot rate
plt.close('all')
for n in ['diff', 'diff_c']:
    if hasattr(conf, n): plt.semilogy(getattr(conf, n))