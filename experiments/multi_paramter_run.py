import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#%%
seed = 492103962
np.random.seed(seed)
#%%
problem = 'Simplex'
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
sweeps = {
  'dt':np.linspace(0,10, 20),
  'sigma': np.linspace(0,10, 10),
  }
#%%
f = conf.get_objective()
x = conf.init_x()
#%%

sweep_keys = sweeps.keys()
ps = list(product(*sweeps.values()))
res = {}
for i,p in enumerate(ps):
    conf.reset_eval()
    kwargs = conf.dyn_kwargs.copy()
    for j, k in enumerate(sweep_keys):
        kwargs[k] = p[j]
    kwargs['M'] = 5
    kwargs['seed'] = seed
    
    dyn = conf.dyn_cls(f, x=x, **kwargs)
    sched = conf.get_scheduler()
    dyn.optimize(sched=sched)
    
    # Evaluate experiment
    conf.evaluate_dynamic(dyn)
    res[p] = conf.diff_c[-1]
    
    if i < len(ps) - 1:
        del dyn
        
#%% plot across dims
plt.close('all')
plot_keys = ['dt','sigma']
fig, ax = plt.subplots(figsize=(6,6))
A = np.zeros(shape=[len(sweeps[k]) for k in plot_keys])
for i,p1 in enumerate(sweeps[plot_keys[0]][::-1]):
    for j,p2 in enumerate(sweeps[plot_keys[1]][::-1]):
        A[i,j] = res[(p1,p2)]
        
extent = [sweeps[plot_keys[1]].min(), sweeps[plot_keys[1]].max(),
          sweeps[plot_keys[0]].min(), sweeps[plot_keys[0]].max()]

im = ax.imshow(A, interpolation="none", 
               extent=extent,
               aspect=extent[1]/extent[3],
               #norm=LogNorm(vmin=0.01, vmax=1)
               )
plt.colorbar(im)
ax.set_xlabel(plot_keys[1])
ax.set_ylabel(plot_keys[0])
plt.tight_layout(pad=0.)
#%%
path = conf.path + 'results/' + conf.name
np.savetxt(path + '_multi_param.txt', A)
plt.savefig(path + 'multi_param.png')

        
