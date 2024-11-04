import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
from cbx.plotting import contour_2D
#%%
np.random.seed(2536309824)
#%%
problem = ''
experiment_name = 'NormSphere_Experiment'#'Ackley_Experiment' 
params  = 'mirror_params_vis'

#%%
CFG = getattr(
    importlib.import_module(
        problem.replace(':','.') + 'experiment_setup'
    ),
    experiment_name
)

param_path = './params/' + params + '.yaml'
conf = CFG(param_path)
save_conf_to_table(conf)
#%%
f = conf.get_objective()
x = conf.init_x()

dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
sched = conf.get_scheduler()
dyn.optimize(sched=sched)
#%%
x, y = [np.array(dyn.history[k]) for k in ['x', 'y']]
plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(15,7))
l = dyn.mirrormap.lamda

mm_kwargs = {'color':'b', 'linestyle' : ':', 'linewidth':2}
for xx in [1, -1]:
    for k in[1,-1]: ax[1].plot(*[[xx, xx], [-4, 4]][::k], **mm_kwargs)
for k in [1,-1]: ax[0].plot(*[[0, 0], [-4, 4]][::k], **mm_kwargs)
ax[1].plot([1.5, 1.5], [-4, 4], color='k', linestyle='--', alpha=0.9, linewidth=2)


for k, z in enumerate([x, y]):
    contour_2D(f, ax=ax[k], num_pts=40, cmap=plt.cm.terrain, 
                            antialiased=False, levels=60, 
                            x_max=4, x_min=-2,alpha=0.7)
    num_pcs = 10
    colors = iter(plt.cm.Reds(np.linspace(0.5,1.,num_pcs)))
    for j in range(num_pcs):
        c = next(colors)
        ax[k].plot(*[z[:, 0, j, i] for i in [0,1]], color=c, linewidth=2., rasterized=False)
        ax[k].scatter(*[z[-1, 0, j, i] for i in [0,1]], color='w', 
                      rasterized=False, zorder=10)
        ax[k].scatter(*[z[0, 0, j, i] for i in [0,1]], color=c, rasterized=False)
    
    ax[k].axis('square')
    ax[k].set_xlim([-1.5,3.5])
    ax[k].set_ylim([-2,2.])
    
plt.tight_layout(pad=0.)
plt.savefig('results/norm_sphere_mirror.pdf',)
    



