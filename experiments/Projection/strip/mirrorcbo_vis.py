from mirrorcbx import MirrorCBO
from exp_setup import ball_strip_experiment
from mirrorcbx.plotting import PlotMirrorDynamicHistory
import numpy as np
import matplotlib.pyplot as plt
from cbx.scheduler import effective_sample_size, multiply
import scienceplots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
#%%
E = ball_strip_experiment(r_min=2, r_max=6, size=(20, 50, 2), x_min=1)

dyn = MirrorCBO(
    E.get_objective(),
    f_dim='3D',
    mirrormap={'name':'ProjectionBall',},
    dt = 0.01,
    sigma= .5,
    noise = 'isotropic',
    alpha = 1.,
    verbosity=0,
    x=E.x_init,  max_it = 4000, 
    track_args={'names':['x', 'drift', 'y', 'consensus']})

sched = multiply(factor=1.05, maximum=1e5)
dyn.optimize(sched=sched)
#%%
E.eval_statistics(dyn)
plt.close('all')

plt.fill_between(E.times, np.min(E.diff, axis=-1),np.max(E.diff, axis=-1),color='orange', alpha=0.5)
plt.semilogy(E.times, E.diff_mean)

#%%
y = np.array(dyn.history['y'])
mm = dyn.mirrormap
BD = mm.Bregman_distance(np.zeros((1,1,1,2)), mm.grad_conj(y))
BD_int = BD.sum(axis=-1)


mass_in_B = ((np.linalg.norm(y,axis=-1) < 1).sum(axis=-1)/y.shape[-2]).mean(axis=-1)

#%%
fig, ax = plt.subplots(1,1, figsize=(7,4))
t = dyn.dt * np.arange(dyn.it)

BD_int_mean = BD_int.mean(axis=-1)
ax.loglog(t, BD_int_mean, color='xkcd:royal blue', linewidth=2.)

bounds = np.linspace(0,1,5)
for b in np.linspace(0, 1, 5):
    idx = np.min(np.where(mass_in_B>=b))
    ax.axvline(x=t[idx], color='k', linestyle='--',alpha=0.7)
    ax.annotate(str(b * 100) + '%', 
                xy=(t[idx], 1), 
                xytext=(t[idx]+0.1, 0.01),
                rotation=90)
    
ix = ax.imshow(mass_in_B[None,:], aspect='auto', cmap=plt.cm.summer, 
          extent=[0.1,np.max(t),np.min(BD_int_mean)*0.5, 50],
          alpha = .5)
plt.colorbar(ix, label='Particles inside the ball')

#for i in range(len(ax)): 
ax.set_xlabel('Time t')
ax.set_ylabel('Bregman distance')
plt.tight_layout(pad=0)
plt.savefig('BregOverMass.png',)


#%%
from matplotlib.patches import Circle
vis_dyn = False
if vis_dyn:
    p = PlotMirrorDynamicHistory(dyn, 
                            objective_args={'x_min':-2, 'x_max':2, 
                                            'num_pts':100, 'levels':50}, 
                            drift_args={'width':0.003},
                            plot_drift=True,
                            num_run = E.bad_idx)
    p.phx.ax.add_patch(Circle((0,0), 1, fill=False, color='r'))
    p.phy.ax.add_patch(Circle((0,0), 1, fill=False, color='r'))
    p.run_plots(wait=0.75, freq=1)



