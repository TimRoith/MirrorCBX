from cbx.dynamics.cbo import CBO, cbo_update
from cbx.dynamics.polarcbo import PolarCBO
from cbx.utils.history import track
from mirrorcbx.mirrormaps import get_mirror_map
import numpy as np

#%%
class track_y(track):
    @staticmethod
    def init_history(dyn):
        dyn.history['y'] = []
    
    @staticmethod
    def update(dyn) -> None:
        dyn.history['y'].append(dyn.copy(dyn.y))
    
    
def mirror_step(self,):
    self.compute_consensus() # compute consensus, sets self.energy and self.consensus
    self.drift = self.correction(self.x[self.particle_idx] - self.consensus) # update drift and apply drift correction
    self.y[self.particle_idx] += cbo_update( # perform cbo update step
        self.drift, self.lamda, self.dt, 
        self.sigma, self.noise()
    )
    self.x = self.mirrormap.grad_conj(self.y)
  
def select_mirrormap(self, mirrormap):
    self.mirrormap = get_mirror_map(mirrormap)
    

class MirrorCBO(CBO):
    def __init__(self, f, mirrormap=None, **kwargs):
        super().__init__(f, **kwargs)
        self.select_mirrormap(mirrormap)
        self.y = self.mirrormap.grad(self.copy(self.x))
        
    select_mirrormap = select_mirrormap
    inner_step = mirror_step
        
    known_tracks = {
        'y': track_y,
        **CBO.known_tracks,}
    
class KickingMirrorCBO(MirrorCBO):
    def __init__(self, f, reg_lamda=1., kick_thresh=1e-10, **kwargs):
        super().__init__(f, **kwargs)
        self.reg_lamda = reg_lamda
        self.kick_thresh = kick_thresh
        
    def inner_step(self,):
        self.compute_consensus() # compute consensus, sets self.energy and self.consensus
        self.drift = self.correction(self.x[self.particle_idx] - self.consensus) # update drift and apply drift correction
        self.kicking_step()
        self.y[self.particle_idx] += cbo_update( # perform cbo update step
            self.drift, self.lamda, self.dt, 
            self.sigma, self.noise()
        )
        self.x = self.mirrormap.grad_conj(self.y)
        
      
    def kicking_step(self,):
        if self.it < 2:
            return
        x, x_old = (self.x[self.particle_idx], self.x_old[self.particle_idx])
        kick_idx = np.linalg.norm(x - x_old, axis=-1) < self.kick_thresh
        zero_idx = np.isclose(x, 0)
        self.kicking_factor = np.ones_like(self.drift)
        
        d_idx = np.abs(self.drift) > 1e-12
        self.kicking_factor[d_idx] += np.ceil(
            (self.reg_lamda * np.sign(self.drift[d_idx]) - self.y[d_idx]) /
            self.drift[d_idx]
            )
        self.kicking_factor = np.int64(np.clip(self.kicking_factor, a_min=1, a_max=8))
        self.kicking_factor[:] = np.min(self.kicking_factor + 8 * ~zero_idx,axis=-1, keepdims=True)[:]
        #self.kicking_factor *= zero_idx
        self.kicking_factor[~kick_idx, :] = 1
        
        self.drift *= self.kicking_factor

    
class PolarMirrorCBO(PolarCBO):
    def __init__(self, f, mirrormap=None, **kwargs):
        super().__init__(f, **kwargs)
        self.select_mirrormap(mirrormap)
        self.y = self.mirrormap.grad(self.copy(self.x))
        
    select_mirrormap = select_mirrormap
    inner_step = mirror_step
        
    known_tracks = {
        'y': track_y,
        **CBO.known_tracks,}