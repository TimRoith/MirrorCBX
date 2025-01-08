import numpy as np
import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
from cbx.plotting import contour_2D
import scienceplots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
#%%
np.random.seed(253630824)
#%%
problem = ''
params  = 'mirror_params_vis2'

#%%
CFG = getattr(
    importlib.import_module(
        problem.replace(':','.') + 'experiment_setup'
    ),
    'select_experiment'
)

param_path = './params/' + params + '.yaml'
conf = CFG(param_path)
#%%
f = conf.get_objective()
x = conf.init_x()

dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
sched = conf.get_scheduler()
dyn.optimize(sched=sched)
#%%
x, y = [np.array(dyn.history[k]) for k in ['x', 'y']]
plt.close('all')
fig, ax = plt.subplots(1,2,figsize=(11.69,8.27))
l = dyn.mirrormap.lamda

idx = np.argmax(np.abs((conf.get_minimizer() - dyn.x)).mean(axis=(-2,-1)))
mm_kwargs = {'color':'w', 'linestyle' : ':', 'linewidth':2}
for xx in [1, -1]:
    for k in[1,-1]: ax[1].plot(*[[xx, xx], [-4, 4]][::k], **mm_kwargs)
for k in [1,-1]: ax[0].plot(*[[0, 0], [-4, 4]][::k], **mm_kwargs)
ax[1].plot([1.5, 1.5], [-4, 4], color='k', linestyle='--', alpha=0.9, linewidth=2)


for k, z in enumerate([x, y]):
    contour_2D(f, ax=ax[k], num_pts=40, cmap=plt.cm.terrain, 
                            antialiased=False, levels=80, 
                            x_max=4, x_min=-2,alpha=0.7)
    ax[k].plot(np.linspace(-3, 3), conf.f - conf.A[0,0]*np.linspace(-3, 3), color='b', 
               linewidth=2.5)
    num_pcs = 10
    colors = iter(plt.cm.Reds(np.linspace(0.5,1.,num_pcs)))
    for j in range(num_pcs):
        c = next(colors)
        ax[k].plot(*[z[:, idx, j, i] for i in [0,1]], color=c, linewidth=3., rasterized=False)
        ax[k].scatter(*[z[-1, idx, j, i] for i in [0,1]], color='w', 
                      rasterized=False, zorder=10, s=70)
        ax[k].scatter(*[z[0, idx, j, i] for i in [0,1]], color=c, rasterized=False, s=100)
    
    ax[k].axis('square')
ax[0].set_xlim([.25,.75])
ax[0].set_ylim([-0.25,0.25])
ax[1].set_xlim([.5,2.5])
ax[1].set_ylim([-.5,1.5])
    
for k, n in enumerate(['Primal Space', 'Dual Space']): ax[k].set_title(n)
plt.tight_layout(pad=.4)
plt.savefig('results/norm_sphere_mirror_failure.pdf',)