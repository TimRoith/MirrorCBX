import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
np.random.seed(1524023)
import numpy as np
import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
import cbx.utils.success as scc
#%%
from experiment_setup import Ackley_Experiment
conf = Ackley_Experiment('params/mirror_params_vis.yaml')
#%%
f = conf.get_objective()
const_minimizer = conf.get_minimizer()
x = conf.init_x()

dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
dyn.optimize(sched=conf.get_scheduler())
#%% Evaluate experiment
fname = conf.config.path+conf.config.name
x = np.array(dyn.history['x'])[1:, ...]
y = np.array(dyn.history['y'])
diff = np.linalg.norm(x - const_minimizer, axis=-1).mean(axis=(-2,-1))
np.savetxt(fname + '_diff.txt', diff)

tol = 0.1
scc_eval = scc.dist_to_min_success(dyn.consensus, const_minimizer, tol=tol)
print('Success rate: ' + str(scc_eval['rate'] ))

#%% plot rate
plt.close('all')
plt.semilogy(diff)


#%%
plt.close('all')
from sphere_utils import grid_x_sph

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
ay = fig.add_subplot(122, projection='3d', computed_zorder=False)

X = grid_x_sph(100000, theta_max=np.pi/2)

f = dyn.f(X)
f = f - np.min(f)
f = f/np.max(f)

for axx, alpha in [(ax, 0.75), (ay, 0.4)]:
    axx.plot_surface(
        *[X[...,i] for i in range(3)],
        shade=False,
        edgecolor='none',
        linewidth=1.,
        #edgealpha=0.4,
        facecolors=plt.cm.terrain(f),
        alpha=alpha,zorder=1,
        rstride=1, cstride=1,
        antialiased=True,
        rasterized=True
    )

ptcl_range = range(25,39)
num_ptcl = len(ptcl_range)
colors = iter(plt.cm.Reds(np.linspace(0.5, 1, num_ptcl)))


for j in range(num_ptcl):
    c = next(colors)
    for axx, z, zo in [(ax, x, 10), (ay, y, 0)]:
        axx.plot(*[z[:, 0, j, i] for i in range(3)], color=c, 
                zorder=zo, linewidth=2., rasterized=True)
        axx.scatter(*[z[0:1, 0, j, i] for i in range(3)], s=150, color = c, zorder=zo, marker='.', rasterized=True)
        axx.scatter(*[z[-2:-1, 0, j, i] for i in range(3)], s=150, color = 'b', zorder=zo, rasterized=True)

ay.plot([0,0], [0,0],[0,5], color='k', alpha=0.5, linestyle='--')
for axx in [ax, ay]:
    axx.view_init(30, 35, 0)
    axx.set_zlim(0,1.)
    axx.set_aspect('equal')
    axx.minorticks_off()

#%%
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.savefig('results/mirrorcbo_sphere_vis.pdf')