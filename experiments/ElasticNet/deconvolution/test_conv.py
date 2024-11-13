from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import fftconvolve, correlate
import numpy as np

#%%
x = np.random.uniform(size=(1,1000,100))
k = np.random.uniform(size=(1, 1 ,10))
#kpad = ifftshift(np.pad(k, pad_width=(4, 4)))

#z = ifft(fft(x,norm='backward') * fft(kpad,norm='backward'),norm='backward').real

zz = fftconvolve(x,k, mode='same')

#%% adjoint
z = np.random.uniform(size=x.shape)
kk = np.concatenate([k[...,::-1], np.zeros((1,1,1))],axis=-1)

h = (fftconvolve(x,k, mode='same') * z).sum(axis=-1) - (fftconvolve(z,kk, mode='same') * x).sum(axis=-1)
print(h)
#%%
class convolution_operator:
    def __init__(self, k, size=None, downsampling = 1):
        self.downsampling = downsampling
        self.k = k
        
        
    def __call__(self, x):
        k = self.k[(x.ndim-1)*(None,) + (Ellipsis,)]
        y = fftconvolve(x, k, mode='same', axes=(-1))
        return y#[..., np.arange(0,y.shape[-1], self.downsampling)]
    
    def adjoint(self, y):
        k = self.k[(y.ndim-1)*(None,) + (Ellipsis,)]
        if (k.shape[-1]%2) == 0: # even kernel length
            k = np.concatenate([k[...,::-1], np.zeros(y.ndim * (1,))], axis=-1)
        else:
            k = k[...,::-1]
    
        
        return fftconvolve(y, k, mode='same', axes=(-1))
    
    T = adjoint
    
A = convolution_operator(k[0,0,:])

h = (A(x) * z).sum(axis=-1) - (A.T(z) * x).sum(axis=-1)
print(h)