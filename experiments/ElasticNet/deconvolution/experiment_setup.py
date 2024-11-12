from mirrorcbx.utils import ExperimentConfig
import numpy as np
import cbx
from cbx.scheduler import scheduler
from numpy.fft import fft, ifft, fftshift, ifftshift
from mirrorcbx.objectives import data_fidelity
from mirrorcbx.regularization import regularize_objective
from scipy.signal import fftconvolve
#%%
def select_experiment(conf_path):
    return Deconvolution_Experiment(conf_path)

#%%
class noise_lvl_pp:
    def __init__(self, thresh=1e-4, loss_thresh=0.1, 
                 indep_sigma=0.01, patience = 20,
                 reset_alpha=1e7, var_name='y',
                 update_thresh=1e-3,
                 decrease_sigma=.99):
        self.thresh = thresh
        self.loss_thresh = loss_thresh
        self.indep_sigma = indep_sigma
        self.best_energy = None
        self.patience = patience
        self.reset_alpha = reset_alpha
        self.energies = []
        self.min_decrease = 0.99
        self.var_name = var_name
        self.update_thresh = update_thresh
        self.decrease_sigma = decrease_sigma
        
    def __call__(self, dyn):
        wt = self.check_consensus_update(dyn)
        #wl = self.check_loss(dyn)
        idx   = np.where(
            wt
            #wl
        )
        var = getattr(dyn, self.var_name)
        if len(idx[0]) > 0:
            z = np.random.normal(0, 1, size = (len(idx[0]), dyn.N) + dyn.d)
            var[idx[0], ...] += self.indep_sigma * (dyn.dt**0.5) * z
            dyn.alpha[idx[0],...] = np.minimum(dyn.alpha[idx[0],...], 
                                               self.reset_alpha
                                               )
            self.indep_sigma *= self.decrease_sigma
        setattr(dyn, self.var_name, np.clip(var, a_min=-100, a_max=100))
        
    
    def check_loss(self, dyn):
        wl = (dyn.energy.mean(axis=-1) > self.loss_thresh)
        return wl
        
    def check_energy(self, dyn):
        e = dyn.energy.min(axis=-1).copy()
        self.energies.append(e)
        self.energies = self.energies[-self.patience:]
        
        if len(self.energies) == self.patience:
            wt = (self.energies[-1]/self.energies[0]) < self.min_decrease
            self.energies = []
        else:
            wt = np.zeros(e.shape[0], dtype=bool)
        
        return wt
    
    def check_consensus_update(self, dyn):
        if not hasattr(self, 'consensus_updates'): self.consensus_updates = []
        wt = np.zeros((dyn.M,), dtype=bool)
        if hasattr(self, 'consensus_old'):
            self.consensus_updates.append(
                np.linalg.norm(dyn.consensus - self.consensus_old)
            )
            if len(self.consensus_updates) == self.patience:
                wt = np.array([self.consensus_updates]).max(axis=-1) < self.update_thresh
                self.consensus_updates = []
        self.consensus_old = dyn.copy(dyn.consensus)
        return wt
    
#%%
class discrepancy_sched:
    def __init__(self, min_it = 100, 
                 patience=100,
                 loss_thresh = 0.1,
                 decr_incr=None,
                 min_max = None):
        self.min_it = min_it
        self.loss_thresh = loss_thresh
        self.patience = patience
        self.decr_incr = (0.999, 1.01) if decr_incr is None else decr_incr
        self.min_max  = (0., 1e2) if min_max is None else min_max
        
    def update(self, dyn):
        if (hasattr(dyn.f, 'lamda') 
            and (dyn.it > self.min_it) 
            and (dyn.it%self.patience == 0)
            ):
            e = dyn.f.original_func(dyn.consensus)
            wl = (e.mean(axis=-1) > self.loss_thresh)
            dyn.f.lamda[wl, ...]  *=  self.decr_incr[0], 
            dyn.f.lamda[~wl, ...] *=  self.decr_incr[1]
            dyn.f.lamda = np.clip(dyn.f.lamda, 
                                  a_min = self.min_max[0], 
                                  a_max = self.min_max[1]
                                  )
        
#%%
class convolution_operator:
    def __init__(self, k, size=None, downsampling = 1):
        self.downsampling = downsampling
        self.k = k
        
        
    def __call__(self, x):
        k = self.k[(x.ndim-1)*(None,) + (Ellipsis,)]
        y = fftconvolve(x, k, mode='same', axes=(-1))
        return y[..., np.arange(0,y.shape[-1], self.downsampling)]
    
    def adjoint(self, y):
        if self.downsampling:
            y_adj = np.zeros(shape=(y.shape[0], 2*y.shape[1]))
            #y_adj[:, np.arange(1,2*y.shape[1],2)] = y.copy()
            y_adj[:, np.arange(0,2*y.shape[1],2)] = y.copy()
        else:
            y_adj = y
        
        x = np.real(ifft(self.j_fft * fft(y_adj, axis=1), axis=1))
        return x
    

        
#%%
class Deconvolution_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.set_reg()
        
    def set_reg(self,):
        dname = self.config.dyn.name
        if dname == 'MirrorCBO':
            self.dyn_kwargs['mirrormap'] = {
                 'name':'ElasticNet',
                 'lamda':getattr(self.config.mirrormap, 'lamda', 1.)
                 }
            
    def set_problem_kwargs(self,):
        pr = self.config.problem
        self.obj, self.d = (pr.obj, pr.d)
        
        defaults = {'tol': 0.1,
                    'num_signals': 5,
                    'downsampling': 2,
                    'kernel_var': 1/20,
                    'kernel_width': 10,
                    'noise_lvl' : 0.01}
        
        for k in defaults.keys():
            setattr(self, k, getattr(pr, k, defaults[k]))
            
        self.time_disc_x    = np.linspace(0, 1, self.d)
        self.time_disc_data = np.linspace(0, 1, self.d//self.downsampling)
        
        self.kernel = np.exp(-(
            np.linspace(-1,1,self.kernel_width))**2 / (2 * self.kernel_var)
            )
        self.A = convolution_operator(
            self.kernel, 
            size = self.d, 
            downsampling = self.downsampling
        )
  
    def set_post_process(self,):
        self.loss_thresh = 0.5 * (self.d//self.downsampling) * self.noise_lvl**2
        pp = getattr(self.config, 'postprocess', {'name':'default'})
        if pp['name'] == 'noise_lvl':
            self.dyn_kwargs['post_process'] = noise_lvl_pp(
                loss_thresh = self.loss_thresh,
                **{k:v for k,v in pp.items() if k not in ['name']}
            )
        
    def get_objective(self,):
        self.set_post_process()
        self.x_true = np.zeros(shape = (self.d,))
        nz_points = np.random.permutation(
            np.arange(self.d)
            )[:self.num_signals]
        self.x_true[nz_points] = np.random.uniform(0,1,size=(self.num_signals,))

        self.y = self.A(self.x_true)
        self.y += self.noise_lvl * np.random.normal(0, 1, size = self.y.shape)

        ob = data_fidelity(self.y, self.A)
        
        reg = getattr(self.config, 'reg', None)
        lamda = 0.0 if reg is None else getattr(reg, 'plamda', 0.0)
        if self.config.dyn.name in ['PenalizedCBO', 'MirrorCBO'] and lamda > 0:
            ob = regularize_objective(
                ob, 
                {'name':getattr(reg, 'name', 'L0')},
                lamda=lamda,)
        return ob
    
    def get_scheduler(self,):
        sched = super().get_scheduler()
        lamda_sched = getattr(self.config, 'lamda_scheduler', {})
        return scheduler([discrepancy_sched(loss_thresh=self.loss_thresh,
                                            **lamda_sched),
                          sched])
    
    def get_minimizer(self,):
        return self.x_true[None, None, :]