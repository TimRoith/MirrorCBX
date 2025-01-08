from cbx.objectives import Quadratic
import numpy as np

#%%

def init_ball_strip(r_min=2., r_max = 3, x_min = 1., size=(1,1,1)):
    x = np.zeros((size[0] * size[1], size[2]))
    l = 0
    
    while l < x.shape[0]:
        z = np.random.normal(0,r_max**2, size=(x.shape[0] - l, x.shape[1]))
        nx = np.linalg.norm(z, axis=-1)
    
        idx, = np.where((z[..., 0] > x_min) * 
                        (nx > r_min) * (nx < r_max)) 
        x[l:l+len(idx), :] = z[idx, :]
        l += len(idx)
    return x.reshape(size)

class ball_strip_experiment:
    def __init__(self, p=1., r_min=2., r_max=3., size = (100,50,2), x_min = 1):
        self.minimum = 0#np.array([self.center[0]- self.radius,0.])
        self.x_init = init_ball_strip(size=size, r_min=r_min, r_max=r_max, x_min=x_min)
        self.p = p
    
    def get_objective(self,):
        return Quadratic()
      
    def eval_statistics(self, dyn):
        c = np.array(dyn.history['consensus']).squeeze()
        self.diff = np.linalg.norm(c - self.minimum, axis=-1)
        self.bad_idx = np.argmax(self.diff[-1,:])

        self.times = dyn.dt* np.arange(dyn.it)
        self.diff_mean = self.diff.mean(axis=-1)
        
    

