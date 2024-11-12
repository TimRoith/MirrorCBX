from cbx.dynamics import CBO
from cbx.plotting import PlotDynamicHistory
from mirrorcbx.regularization import regularize_objective
from exp_setup import norm_sphere_experiment
import numpy as np
import matplotlib.pyplot as plt
from cbx.scheduler import effective_sample_size
#%%
E = norm_sphere_experiment()

f = regularize_objective(E.get_objective(), lamda = 0.2, reg_func='L1')
dyn = CBO(
    f,
    f_dim = '3D',
    dt = 0.1,
    sigma= 1.0,
    noise = 'isotropic',
    alpha = 10.,
    verbosity=0,
    x=E.x_init,  max_it = 300, 
    track_args={'names':['x', 'drift', 'consensus']})

sched = effective_sample_size(eta=.5)
dyn.optimize(sched=sched)


#%%
E.eval_statistics(dyn)
plt.close('all')

plt.fill_between(E.times, np.min(E.diff, axis=-1),np.max(E.diff, axis=-1),color='orange', alpha=0.5)
plt.semilogy(E.times, E.diff_mean)
#%%
p = PlotDynamicHistory(dyn, 
                       objective_args={'x_min':-2, 'x_max':2, 
                                       'num_pts':100, 'levels':50}, 
                       drift_args={'width':0.003},
                       plot_drift=True)
p.run_plots(wait=0.5, freq=5)



