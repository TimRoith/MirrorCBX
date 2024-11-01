import numpy as np
import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
#%%
problem = 'Hypersurfaces:Sphere'
experiment_name = 'PhaseRetrieval_Experiment'#'Ackley_Experiment' 
params  = 'mirror_params_phaseN0'

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
for reps in range(getattr(conf.config, 'reps', 1)):
    f = conf.get_objective()
    const_minimizer = conf.get_minimizer()
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