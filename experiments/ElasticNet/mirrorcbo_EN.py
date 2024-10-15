from mirrorcbx import MirrorCBO
from mirrorcbx.objectives import norm_sphere
from cbx.plotting import PlotDynamicHistory
import numpy as np
import matplotlib.pyplot as plt
#%%
f = norm_sphere(center= np.array([1.5,0.])[None,None,:])

x = np.random.uniform(low=-3,high=3., size=(1,100,2))
dyn = MirrorCBO(
    f,
    mirrormap={'name':'ElasticNet', 'lamda':1.0},
    dt = 0.1,
    sigma= 1.0,
    noise = 'isotropic',
    alpha = 10.,
    x=x,  max_it = 300, 
    track_args={'names':['x', 'drift']})

#%%
dyn.optimize()
#%%
plt.close('all')
p = PlotDynamicHistory(dyn, 
                       objective_args={'x_min':-2, 'x_max':2, 
                                       'num_pts':100, 'levels':50}, 
                       drift_args={'width':0.003},
                       plot_drift=True)
p.run_plots(wait=0.5, freq=5.)



