import matplotlib.pyplot as plt
from pkgutil import importlib
import numpy as np
#%%
np.random.seed(35921724)
#%%
problem = ''
params  = 'reg_vis'

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
if hasattr(dyn.f, 'original_func'):
    e  = dyn.f.original_func(np.array(dyn.history['consensus'])).mean(axis=-1).squeeze()
    e2 = dyn.f(np.array(dyn.history['consensus'])).mean(axis=-1).squeeze()
else:
    e = np.array(dyn.history['energy']).mean(axis=-1).squeeze()
    e2 = e
    
ec = np.convolve(e, np.ones(200), 'same') / 200

ax[0].semilogy(e, color='orange')
ax[0].semilogy(e2, color='blue')
ax[0].semilogy(ec, color='red', linestyle='dashed')
ax[0].plot([0,dyn.it], 2*[conf.loss_thresh], linestyle='dashed')
#
ax[2].stem(conf.time_disc_x, conf.x_true, linefmt='blue')
idx = -1#np.argmax(e < conf.loss_thresh)

c = np.array(dyn.history['consensus'])
ax[2].stem(conf.time_disc_x, c[idx, ...].squeeze(), linefmt='green', markerfmt='D')
ax[3].stem(conf.time_disc_data, conf.y.squeeze(), linefmt='red')
ax[3].plot(conf.time_disc_data, conf.y.squeeze(), color='red')
yy = dyn.f.original_func.A(dyn.consensus).squeeze()
ax[3].stem(conf.time_disc_data, yy, linefmt='y')
ax[3].plot(conf.time_disc_data, yy, color='y')
ax[3].stem(conf.time_disc_x, conf.x_true, linefmt='blue')

#%%
for zz, n in [(dyn.consensus, 'deconv_result_c'),
              (conf.y, 'deconv_result_meas'),
              (conf.x_true, 'deconv_result_x')]:
    np.savetxt(conf.path + 'results/' + n + '.txt', zz.squeeze())
