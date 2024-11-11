import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
from pkgutil import importlib
import numpy as np
#%%
np.random.seed(35253324)
#%%
problem = ''
params  = 'mirror_vis2'

#%%
CFG = getattr(
    importlib.import_module(
        problem.replace(':','.') + 'experiment_setup'
    ),
    'select_experiment'
)

param_path = problem.replace(':','/') + './params/' + params + '.yaml'
conf = CFG(param_path)
#%%
f = conf.get_objective()
x = conf.init_x()

dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
sched = conf.get_scheduler()
dyn.optimize(sched=sched)

# Evaluate experiment
conf.evaluate_dynamic(dyn)

#%% plot rate
plt.close('all')
fig, ax = plt.subplots(1,4)
ax[0].semilogy(conf.diff_c, color='green')
ax[0].semilogy(np.array(dyn.history['energy']).min(axis=-1).squeeze(), color='orange')
#ax[0].semilogy(np.array(dyn.history['alpha']).squeeze())
ax[0].plot([0,dyn.it], 2*[conf.loss_thresh], linestyle='dashed')
#
ax[2].stem(conf.time_disc_x, conf.x_true, linefmt='blue')
ax[2].stem(conf.time_disc_x, dyn.consensus.squeeze(), linefmt='green', markerfmt='D')
c = np.array(dyn.history['consensus'])
#ax[2].stem(conf.time_disc_x, c[150].squeeze(), linefmt='red', markerfmt='D')
ax[3].stem(conf.time_disc_data, conf.y.squeeze(), linefmt='red')
ax[3].plot(conf.time_disc_data, conf.y.squeeze(), color='red')
ax[3].stem(conf.time_disc_x, conf.x_true, linefmt='blue')
