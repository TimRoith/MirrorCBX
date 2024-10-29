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