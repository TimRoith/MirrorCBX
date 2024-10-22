from cbx.objectives import Rastrigin
import numpy as np

#%%
class Rastrigin_experiment:
    def __init__(self, p=1.):
        self.b = 0.75
        self.minimum = np.array([self.b, self.b])
        self.x_init = np.random.uniform(low=-1,high=1., size=(200,150,2))
    
    def get_objective(self,):
        return Rastrigin(b=self.b)
      
    def eval_statistics(self, dyn):
        c = np.array(dyn.history['consensus']).squeeze()
        self.diff = np.linalg.norm(c - self.minimum, axis=-1)
        self.bad_idx = np.argmax(self.diff[-1,:])

        self.times = dyn.dt* np.arange(dyn.it)
        self.diff_mean = self.diff.mean(axis=-1)
        
    

