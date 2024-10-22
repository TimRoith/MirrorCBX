from mirrorcbx import MirrorCBO
from exp_setup import Rastrigin_experiment
from mirrorcbx.plotting import PlotMirrorDynamicHistory
import numpy as np
import matplotlib.pyplot as plt
from cbx.scheduler import effective_sample_size, multiply
#%%
E = Rastrigin_experiment()

dyn = MirrorCBO(
    E.get_objective(),
    f_dim='3D',
    mirrormap={'name':'LogBarrierBox',},
    dt = 0.01,
    sigma= 1.5,
    noise = 'isotropic',
    alpha = 10.,
    verbosity=0,
    x=E.x_init,  max_it = 300, 
    track_args={'names':['x', 'drift', 'y', 'consensus']})

sched = multiply(factor=1.05, maximum=1e12)
dyn.optimize(sched=sched)
#%%
E.eval_statistics(dyn)
plt.close('all')

plt.fill_between(E.times, np.min(E.diff, axis=-1),np.max(E.diff, axis=-1),color='orange', alpha=0.5)
plt.semilogy(E.times, E.diff_mean)
#%%
from matplotlib.patches import Rectangle
p = PlotMirrorDynamicHistory(dyn, 
                        objective_args={'x_min':-2, 'x_max':2, 
                                        'num_pts':100, 'levels':50}, 
                        drift_args={'width':0.003},
                        plot_drift=True,
                        num_run = E.bad_idx)
p.phx.ax.add_patch(Rectangle((-1,-1), 2, 2, fill=False))
p.run_plots(wait=0.1, freq=1)



