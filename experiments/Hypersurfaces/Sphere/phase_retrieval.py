import numpy as np

#%%
class operator:
    def __init__(self, f):
        self.f = f
        
    def __call__(self, x):
        return (x@self.f.T)**2 #np.abs(np.einsum('ki,...i->...k', self.f, x))**2
    

def lower_frame_bound(f):
    '''
    This function is a variation of the function here: 
        https://github.com/PhilippeSu/KV-CBO/blob/master/v1/LowerFrameBound.m

    Parameters
    ----------
    f : array, (M,d)
        A frame of M vectors for the space Rd

    Returns
    -------
    A:  float
        lower frame bound for frame f

    '''

    #l, _ = np.linalg.eig(f.T@f)
    _, D, _ = np.linalg.svd(f)
    return np.min(D)**2
    
    
    
class objective:
    def __init__(self, y, f):
        self.y = y
        self.operator = operator(f)
        self.setR(f)
        
    def __call__(self, x):
        Fx = self.operator(x[..., :-1])
        return np.linalg.norm(Fx - self.y/(self.R**2), axis=-1)**2
        
    def setR(self, f):
        self.A = lower_frame_bound(f)
        self.R = ((1/self.A) * np.sum(self.y))**0.5
        
def get_error_min(x_true, x):
    return np.min(
        np.array([
            np.linalg.norm(x_true - a*x, axis=-1, ord=float('inf')) 
            for a in [-1, 1]
            ]), 
        axis=0
    )



def WirtingerFlow(f, y, z0=None, max_it=100, muf=1., tau_0=300, mu_max=0.4):
    z = spectral_init(f,y) if z0 is None else z0.copy()
    nz0 = np.linalg.norm(z)**2
    M = f.shape[0]
    
    for i in range(max_it):
        mu = muf * min(1 - np.exp(-i/tau_0), mu_max)
        g = ((f@z)**2 - y)[:, None] * ((f[:,None,:] * f[...,None])@z)
        z = z - (mu/nz0) * g.mean(axis=0)
        
        
        print((((f@z)**2 - y)**2).sum())
    return z

def energy(f, y, z):
    return (((f@z)**2 - y)**2).sum()


def WirtingerFlowBackTracking(f, y, z0=None, max_it=100, mu0=10.,
                              verbosity=0,
                              energy_tol=1e-8):
    z = spectral_init(f,y) if z0 is None else z0.copy()
    nz0 = np.linalg.norm(z)**2
    M = f.shape[0]
    mu0 = mu0
    
    for i in range(max_it):
        g = ((f@z)**2 - y)[:, None] * ((f[:,None,:] * f[...,None])@z)
        g = g.mean(axis=0)
        
        energy_old = energy(f, y, z)
        
        mu = mu0
        for bc in range(50):
            tmp = energy_old + 0.1 * mu * np.linalg.norm(g)**2
            
            z_new = z - mu * g
            energy_new = energy(f, y, z_new)
            
            if energy_new <= tmp:
                break
            mu = mu0 * 0.2
        
        z = z_new.copy()
        
        if verbosity > 0:
            print(energy_new)
        if energy_new < energy_tol:
            break
        
    return z


def spectral_init(f, y):
    m,n = f.shape
    lamda = np.sqrt(n * y.sum()/(np.linalg.norm(f.ravel())**2))
    
    Y = (y[:, None,None] * (f[:,None,:] * f[...,None])).mean(axis=0)
    l, v = np.linalg.eig(Y)
    idx = np.argmax(l)
    return v[:, idx] * lamda
    
    