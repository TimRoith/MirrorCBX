from mirrorcbx.dynamics import SphereCBO
from mirrorcbx.plotting import PlotMirrorDynamicHistory
import numpy as np
import matplotlib.pyplot as plt

from cbx.objectives import Ackley
#%%
d = 2
f = Ackley(minimum=0.4*np.ones((1,1,d)), b=0.1)
x = np.random.normal(0,1, (100, 50, d))
x[..., -1] *= np.sign(x[..., -1])


x = x/np.linalg.norm(x, axis=-1, keepdims=True)

const_minimizer = 1/(d**0.5) * np.ones((d,))


dyn = SphereCBO(
    f,
    f_dim='3D',
    dt = 0.05,
    sigma= .25,
    noise = 'isotropic',
    alpha = 50.,
    verbosity=0,
    x=x,  max_it = 400, 
    track_args={'names':['x', 'drift', 'consensus']})


dyn.optimize()
#%%
plt.figure()
x = np.array(dyn.history['x'])
diff = np.linalg.norm(x - const_minimizer, axis=-1).mean(axis=(-2,-1))
np.save('results/spherecbo.npy', diff)
plt.semilogy(diff)



#%%
from matplotlib.patches import Circle
from cbx.plotting import PlotDynamicHistory
vis_dyn = True
if vis_dyn:
    p = PlotDynamicHistory(dyn, 
                            objective_args={'x_min':-2, 'x_max':2, 
                                            'num_pts':100, 'levels':50}, 
                            drift_args={'width':0.003},
                            plot_drift=True,
                            num_run = 0)
    p.ax.add_patch(Circle((0,0), 1, fill=False, color='r'))
    p.run_plots(wait=0.5, freq=5)



