import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

#%% custom imports
from inspect import getsourcefile
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])

import mirrorcbo as mcbo
import mirrorcbo.particledynamic as pdyn
import initialization

#%%
cur_path = os.path.dirname(os.path.realpath(__file__))

#%% set parameters
conf = mcbo.utils.config()
conf.save2disk = False
conf.T = 1001
conf.tau=0.01 #timestep
conf.x_max = 4
conf.x_min = -4
conf.random_seed = 42
conf.d = 2
conf.beta = 1
conf.sigma = 1.0
conf.heavy_correction = False
conf.num_particles = 200
conf.factor = 2.0
conf.noise = mcbo.noise.normal_noise(tau=conf.tau)
conf.eta = 0.5


conf.MirrorFct = mcbo.functional.ElasticNet(lamda=1)
# conf.MirrorFct = mcbo.functional.ProjectionBall(radius=1, center=np.array([1,2]))
# conf.MirrorFct = mcbo.functional.LogBarrier()


snapshots = [0, 100, 500, 1000, 2000]


conf.V = mcbo.objectives.Rastrigin()

#%% initialize scheme
np.random.seed(seed=conf.random_seed)
x = mcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                      x_min=conf.x_min, x_max=conf.x_max)
    
#%% init optimizer and scheduler
opt = pdyn.MirrorCBO(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                       beta = conf.beta, MirrorFct = conf.MirrorFct)

beta_sched = mcbo.scheduler.beta_exponential(opt, r=conf.factor, beta_max=1e7)

#%% plot loss landscape and scatter
plt.close('all')
fig, ax = plt.subplots(1,1, squeeze=False)
rc('font',**{'family':'serif','serif':['Times'],'size':14})
rc('text', usetex=True)

colors = ['peru','tab:pink','deeppink', 'steelblue', 'tan', 'sienna',  'olive', 'coral']
num_pts_landscape = 200
xx = np.linspace(conf.x_min, conf.x_max, num_pts_landscape)
yy = np.linspace(conf.x_min,conf.x_max, num_pts_landscape)
XX, YY = np.meshgrid(xx,yy)
XXYY = np.stack((XX.T,YY.T)).T
Z = np.zeros((num_pts_landscape,num_pts_landscape,conf.d))
Z[:,:,0:2] = XXYY
ZZ = opt.V(Z)
lsp = np.linspace(np.min(ZZ),np.max(ZZ),15)
cf = ax[0,0].contourf(XX,YY,ZZ, levels=lsp)
#plt.colorbar(cf)
ax[0,0].axis('square')
ax[0,0].set_xlim(conf.x_min,conf.x_max)
ax[0,0].set_ylim(conf.x_min,conf.x_max)

#plot points and quiver
scx = ax[0,0].scatter(opt.x[:,0], opt.x[:,1], marker='o', color=colors[1], s=12)
# scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[2], s=30)
quiver = ax[0,0].quiver(opt.x[:,0], opt.x[:,1], opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1], color=colors[1], scale=20)
#scm = ax[0,0].scatter(opt.m_beta[:,0], opt.m_beta[:,1], marker='x', color=colors[0], s=50)

time = 0.0
#%% main loop
if conf.save2disk:
    path = cur_path+"\\visualizations\\Rastrigin\\"
    os.makedirs(path, exist_ok=True) 

threshold = 1e-2; rate = 0; N_simul = 10
for n in range(N_simul):
    opt = initialization.init(conf)
    for i in range(conf.T):
        # plot
        if i%100 == 0:
            scx.set_offsets(opt.x[:, 0:2])
            # scm.set_offsets(opt.m_beta[:, 0:2])
            quiver.set_offsets(np.array([opt.x[:,0], opt.x[:,1]]).T)
            quiver.set_UVC(opt.m_beta[:,0]-opt.x[:,0], opt.m_beta[:,1]-opt.x[:,1])
            # plt.title('Time = ' + str(time) + ' beta: ' + str(opt.beta) + ' kappa: ' + str(opt.kernel.kappa))
            if n == 0:
                plt.title('Time = ' + str(time))
                plt.pause(0.1)
            # plt.show()
            
            if conf.save2disk is True and i in snapshots:
                fig.savefig(path+"Rastrigin-i-" \
                            + str(i) + ".pdf",bbox_inches="tight")
                
        
        # update step
        time = conf.tau*(i+1)
        opt.step(time=time)
        beta_sched.update()

    #compute success rate
    val = conf.V(opt.x)
    # maxvalindex = val.argmax(axis = 0)
    # maxval = max(val)
    meanval = sum(val)/val.size
    test1 = meanval < threshold
    if test1 == 1:
        rate = rate + 1
    
rate = rate/N_simul
print("Success rate is ", rate)
quit()
# %%
