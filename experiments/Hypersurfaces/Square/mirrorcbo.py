import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
np.random.seed(1524023)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib ipympl
from mirrorcbx.utils import save_conf_to_table
from mirrorcbx.mirrormaps import ProjectionSquare, MirrorMaptoPostProcessProx
import cbx.utils.success as scc
from mirrorcbx.dynamics import MirrorCBO
from cbx.dynamics import CBO
from cbx.objectives import Rastrigin, Ackley, eggholder, Bukin6, Michalewicz, Holder_table
from cbx.plotting import contour_2D
from matplotlib.patches import Rectangle
from cbx.scheduler import multiply

#%%
f = Holder_table(factor=2, shift = np.array([0.2,0.0]))# eggholder()#Ackley()
const_minimizer = np.zeros((2))
mm = ProjectionSquare()
x = np.random.uniform(-3, 3, (1, 30, 2))
#x = mm.grad_conj(x)


dyn = MirrorCBO(f, x=x, max_it=600, 
                mirrormap=mm,
                alpha=10.,
                track_args = {'names':['x','y', 'energy']},
                sigma=0.05,
                dt = 0.01
                )
dyn.optimize(sched=multiply(factor=1.05, maximum=1e12))

#%%
plt.close('all')
x = np.array(dyn.history['x'])[1:, ...]
y = np.array(dyn.history['y'])
fig, ax = plt.subplots(1,2, figsize=(10,3))

akwargs = {'color':'k', 'linestyle': '--', 'linewidth': 1, 'alpha':0.5}
ax[1].plot([0,0], [-1.5,1.5], **akwargs)
ax[1].plot([-1.5,1.5], [0,0], **akwargs)
ax[1].plot([-1,1], [1,-1], **akwargs)
ax[1].plot([-1,1], [-1,1], **akwargs)


for k, z in enumerate([x,y]):
    contour_2D(f, ax=ax[k], num_pts=40, cmap=plt.cm.terrain, 
                            antialiased=False, levels=60, 
                            x_max=2, x_min=-2, rasterized=True)
    ax[k].add_patch(Rectangle((-1, -1), 2, 2, color='k', fill=False))
    colors = iter(plt.cm.Reds(np.linspace(0.5,1.,10)))
    j, taken = 0, 0
    lim = 1.5
    while taken < 6:
        if np.all(np.abs(y[-1,0,j, :]) < lim):
            c = next(colors)
            ax[k].plot(*[z[:, 0, j, i] for i in [0,1]], color=c, linewidth=2., rasterized=False)
            ax[k].scatter(*[z[-1, 0, j, i] for i in [0,1]], color='b', rasterized=False, zorder=10)
            ax[k].scatter(*[z[0, 0, j, i] for i in [0,1]], color=c, rasterized=False, zorder=10)
            taken += 1
        j+=1
        if j >= y.shape[2]:
            break
    
    ax[k].axis('square')
    
    ax[k].set_xlim([-lim,lim])
    ax[k].set_ylim([-lim,lim])

#%%
plt.tight_layout(pad=0., w_pad=0., h_pad = 0.)
plt.savefig('results/square_vis.pdf')