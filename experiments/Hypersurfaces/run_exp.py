import numpy as np
import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
import cbx.utils.success as scc
#%%
exp = 'Sphere'
if exp == 'Plane':
    from Hyperplane.experiment_setup import Ackley_Experiment
    conf = Ackley_Experiment('Hyperplane/params/driftconst_params.yaml')
    save_conf_to_table(conf.config)
elif exp == 'Sphere':
    from Sphere.experiment_setup import Ackley_Experiment
    conf = Ackley_Experiment('Sphere/params/penalized_params.yaml')
    save_conf_to_table(conf.config)
#%%
f = conf.get_objective()
const_minimizer = conf.get_minimizer()
x = conf.init_x()

dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
dyn.optimize(sched=conf.get_scheduler())
#%% Evaluate experiment
fname = conf.config.path+conf.config.name
x = np.array(dyn.history['x'])[1:, ...]
diff = np.linalg.norm(x - const_minimizer, axis=-1).mean(axis=(-2,-1))
np.savetxt(fname + '_diff.txt', diff)

tol = 0.1
scc_eval = scc.dist_to_min_success(dyn.consensus, const_minimizer, tol=tol)
np.savetxt(fname + '_scc.txt', np.array([scc_eval['rate'], tol]))
print('Success rate: ' + str(scc_eval['rate'] ))

#%% plot rate
plt.close('all')
plt.semilogy(diff)