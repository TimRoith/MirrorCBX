from mirrorcbx.objectives import norm_sphere
import numpy as np

#%%
class norm_sphere_experiment:
    def __init__(self, p=1.):
        self.center = np.array([1.5,0.])
        self.radius = 1.
        self.minimum = np.array([self.center[0]- self.radius,0.])
        self.x_init = np.random.uniform(low=-3,high=3., size=(200,50,2))
        self.p = p
    
    def get_objective(self,):
        return norm_sphere(center = self.center[None,None,:], p = self.p)
      
    def eval_statistics(self, dyn):
        c = np.array(dyn.history['consensus']).squeeze()
        self.diff = np.linalg.norm(c - self.minimum, axis=-1)
        self.bad_idx = np.argmax(self.diff[-1,:])

        self.times = dyn.dt* np.arange(dyn.it)
        self.diff_mean = self.diff.mean(axis=-1)
        
    

