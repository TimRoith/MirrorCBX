from mirrorcbx.utils import ExperimentConfig
import numpy as np
import cbx
from numpy.fft import fft, ifft, fftshift, ifftshift

#%%
def select_experiment(conf_path):
    return Deconvolution_Experiment(conf_path)

#%%
class convolution_operator:
    def __init__(self, k, size=None, downsampling = 1):
        self.downsampling = downsampling
        
        if size is None:
            self.size = len(k)
        else:
            self.size = size
            
        diff = size - len(k)
        left_margin = int(diff / 2)
        if diff % 2 == 0:            
            right_margin = left_margin
        else:
            right_margin = left_margin + 1
                        
        self.k = np.pad(k, (left_margin, right_margin))[None,:]
        self.k_fft = fft(self.k)
        
        
    def __call__(self, x):
        y = np.real(ifftshift(ifft(self.k_fft * fft(x, axis=-1), axis=-1),axes=-1))
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
                 'lamda':getattr(self.config.reg, 'lamda', 1.)
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
            
        self.time_disc_data = np.linspace(0, 1, self.d)
        self.time_disc_data = np.linspace(0, 1, self.d//self.downsampling)
        
        self.kernel = np.exp(-(
            np.linspace(-1,1,self.kernel_width))**2 / (2 * self.kernel_var)
            )
        self.A = convolution_operator(
            self.kernel, 
            size = self.d, 
            downsampling = self.downsampling
        )
        
    def get_objective(self,):
        self.x_true = np.zeros(shape = (self.d,))
        nz_points = np.random.permutation(
            np.arange(self.d)
            )[:self.num_signals]
        self.x_true[nz_points] = np.random.uniform(0,1,size=(self.num_signals,))

        self.y = self.A(self.x_true)
        self.y += self.noise_lvl * np.random.normal(0, 1, size = self.y.shape)

        return DatafidelityLoss(self.y, self.A)
    
    def get_minimizer(self,):
        return self.x_true[None, None, :]