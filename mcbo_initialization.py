import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os

import mirrorcbo as mcbo
import mirrorcbo.particledynamic as pdyn

def init(conf, version):
    
    np.random.seed(seed=conf.random_seed)
    x = mcbo.utils.init_particles(num_particles=conf.num_particles, d=conf.d,\
                        x_min=conf.x_min, x_max=conf.x_max)
        
    #%% init optimizer and scheduler
    if version == 1:
        opt = pdyn.MirrorCBO(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                        beta = conf.beta, MirrorFct = conf.MirrorFct)
    else:
        opt = pdyn.MirrorCBO2(x, conf.V, conf.noise, sigma=conf.sigma, tau=conf.tau,\
                        beta = conf.beta, MirrorFct = conf.MirrorFct)
    beta_sched = mcbo.scheduler.beta_exponential(opt, r=conf.factor, beta_max=1e7)
    return opt