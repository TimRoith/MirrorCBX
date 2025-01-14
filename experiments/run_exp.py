import matplotlib.pyplot as plt
from pkgutil import importlib
import numpy as np
#%%
problem = 'ElasticNet:nn'
params  = 'mirror'

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
for _ in range(conf.num_sweeps):
    conf.set_sweep()
    reps = getattr(conf.config, 'reps', 1)
    for rep in range(reps):
        np.random.seed((3**rep)%((2**32)-1))
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
for n in ['diff', 'diff_c', 'energy']:
    if hasattr(conf, n): plt.semilogy(getattr(conf, n), label=n)
plt.legend()