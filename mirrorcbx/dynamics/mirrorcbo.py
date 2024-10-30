from cbx.dynamics.cbo import CBO, cbo_update
from cbx.dynamics.polarcbo import PolarCBO
from cbx.utils.history import track
from mirrorcbx.mirrormaps import get_mirror_map

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