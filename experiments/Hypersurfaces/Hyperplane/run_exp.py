import numpy as np
import matplotlib.pyplot as plt
from mirrorcbx.utils import save_conf_to_table
import cbx.utils.success as scc
import scienceplots
from experiment_setup import Ackley_Experiment
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
np.random.seed(154023)
#%%
conf = Ackley_Experiment('params/sphere_params.yaml')
save_conf_to_table(conf.config)
#%%
f = conf.get_objective()
const_minimizer = conf.get_minimizer()
x = conf.init_x()

dyn = conf.dyn_cls(f, x=x, **conf.dyn_kwargs)
dyn.optimize(sched=conf.get_scheduler())
#%% Evaluate experiment

plt.figure()
fname = conf.config.path+conf.config.name
x = np.array(dyn.history['x'])[1:, ...]
diff = np.linalg.norm(x - const_minimizer, axis=-1).mean(axis=(-2,-1))
np.savetxt(fname + '_diff.txt', diff)

scc_eval = scc.dist_to_min_success(dyn.consensus, const_minimizer, tol=0.1)
np.savetxt(fname + '_scc.txt', scc_eval['rate'] * np.ones(1,))
print('Success rate: ' + str(scc_eval['rate'] ))

#%% plot rate
plt.close('all')
plt.semilogy(diff)