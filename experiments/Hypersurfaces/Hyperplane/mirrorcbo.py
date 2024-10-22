from mirrorcbx.dynamics import MirrorCBO
from mirrorcbx.utils import save_conf_to_table
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from experiment_setup import Ackley_Experiment
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
np.random.seed(154023)
#%%
conf = Ackley_Experiment('params/mirror_params.yaml')
save_conf_to_table(conf.config)
#%%
f = conf.get_objective()
const_minimizer = conf.get_minimizer()
x = conf.init_x()

dyn = MirrorCBO(f, x=x, **conf.dyn_kwargs)
dyn.optimize(sched=conf.get_scheduler())
#%%
plt.close('all')
plt.figure()
fname = conf.config.path+conf.config.name
x = np.array(dyn.history['x'])[1:, ...]
y = np.array(dyn.history['y'])
diff = np.linalg.norm(x - const_minimizer, axis=-1).mean(axis=(-2,-1))
np.savetxt(fname + '_diff.txt', diff)

plt.semilogy(diff)

#%%
vis_3d = (conf.d==3) and False
if vis_3d:
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ay = fig.add_subplot(122, projection='3d', computed_zorder=False)
    
    mm = dyn.mirrormap
    X, Y = np.meshgrid(*[np.linspace(-2,2, 200) for _ in [0,1]]) 
    X = np.stack([X, Y, (-mm.a[0] * X -mm. a[1] * Y + mm.b) * 1. /mm.a[2]],axis=-1)
    
    f = dyn.f(X)
    f = f - np.min(f)
    f = f/np.max(f)
    
    for axx, alpha in [(ax, 0.6), (ay, 0.5)]:
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
    
    ptcl_range = range(10,19)
    num_ptcl = len(ptcl_range)
    colors = iter(plt.cm.Reds(np.linspace(0.5, 1, num_ptcl)))
    
    # plot trajectories
    for j in ptcl_range:
        c = next(colors)
        for axx, z, zo in [(ax, x, 10), (ay, y, 0)]:
            axx.plot(*[z[:, 0, j, i] for i in range(3)], color=c, 
                    zorder=zo, linewidth=3., rasterized=True)
            axx.scatter(*[z[0:1, 0, j, i] for i in range(3)], s=150, color = c, zorder=zo, rasterized=True)
            axx.scatter(*[z[-2:-1, 0, j, i] for i in range(3)], s=150, color = 'b', zorder=zo, rasterized=True)
    
    l = np.stack([-2*mm.a,2*mm.a])
    ay.plot(*[l[:,i] for i in range(3)], color='k', alpha=0.5, linestyle='--')
    for axx in [ax, ay]:
        axx.view_init(10, -15, 0)
        axx.set_zlim(0,1.)
        axx.set_xlim(-2,2.)
        axx.set_ylim(-2,2.)
        axx.set_aspect('equal')
        
    plt.tight_layout(pad=0., h_pad=0, w_pad=0)
    plt.savefig('results/mirrorcbo_plane_vis.pdf')



