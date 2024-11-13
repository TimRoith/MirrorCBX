import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
#%%
problem = 'ElasticNet:deconvolution'
params  = 'CBO'

#%%
CFG = getattr(
    importlib.import_module(
        problem.replace(':','.') + '.experiment_setup'
    ),
    'select_experiment'
)

param_path = problem.replace(':','/') + '/params/' + params + '.yaml'
conf = CFG(param_path)
delete_dyn = True
#%%
reps = getattr(conf.config, 'reps', 1)
for rep in range(reps):
    f = conf.get_objective()
    x = conf.init_x()
    
    dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
    sched = conf.get_scheduler()
    dyn.optimize(sched=sched)
    
    # Evaluate experiment
    conf.evaluate_dynamic(dyn)
    
    if rep < reps - 1:
        del dyn

#%% plot rate
plt.close('all')
for n in ['diff', 'diff_c']:
    if hasattr(conf, n): plt.semilogy(getattr(conf, n))