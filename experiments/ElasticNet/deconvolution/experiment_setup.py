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
                 reset_alpha=1e7):
        self.thresh = thresh
        self.loss_thresh = loss_thresh
        self.indep_sigma = indep_sigma
        self.best_energy = None
        self.patience = patience
        self.reset_alpha = reset_alpha
        
    def __call__(self, dyn):
        wt = self.check_energy(dyn)
        wl = self.check_loss(dyn)
        idx   = np.where(
            wt * 
            wl
        )
        if len(idx[0]) > 0:
            z = np.random.normal(0, 1, size = (len(idx[0]), dyn.N) + dyn.d)
            dyn.x[idx[0], ...] += self.indep_sigma * z
            dyn.alpha[idx[0],...] = np.minimum(dyn.alpha[idx[0],...], 
                                               self.reset_alpha
                                               )
        dyn.x = np.clip(dyn.x, a_min=-100, a_max=100)
        
    
    def check_loss(self, dyn):
        wl = (dyn.energy.mean(axis=-1) > self.loss_thresh)
        return wl
        
    def check_energy(self, dyn):
        if self.best_energy is None: 
            self.best_energy = float('inf') * np.ones((dyn.M))
            self.wait = np.zeros((dyn.M))
        
        self.wait += 1
        e = dyn.energy.min(axis=-1).copy()
        idx = np.where(self.best_energy > e)
        self.wait[idx] = 0
        self.best_energy[idx] = e
        
        wt = self.wait > self.patience
        self.wait[wt] = 0
        
        return wt
    
#%%
class discrepancy_sched:
    def __init__(self, min_it = 100, patience=10, N=20, 
                 loss_thresh = 0.1):
        self.min_it = min_it
        self.loss_thresh = loss_thresh
        self.patience = patience
        
    def update(self, dyn):
        wl = (dyn.energy.mean(axis=-1) > self.loss_thresh)
        if (hasattr(dyn.f, 'lamda') 
            and (dyn.it > self.min_it) 
            and (dyn.it%self.patience == 0)
            ):
            dyn.f.lamda[wl, ...] *= 0.999
            dyn.f.lamda[~wl, ...] *= 1.01
        
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
        self.set_post_process()
        
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
        self.loss_thresh = 0.5 * self.d//self.downsampling * self.noise_lvl**2
        pp = getattr(self.config, 'postprocess', {'name':'default'})
        if pp['name'] == 'noise_lvl':
            self.dyn_kwargs['post_process'] = noise_lvl_pp(
                loss_thresh = self.loss_thresh,
                **pp.indep_noise,
                patience=pp.patience
            )
        
    def get_objective(self,):
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
        return scheduler([discrepancy_sched(loss_thresh=self.loss_thresh), 
                          sched])
    
    def get_minimizer(self,):
        return self.x_true[None, None, :]