import numpy as np

class norm_sphere:
    def __init__(self, radius=1., p=1, center= None, d=2):
        self.center = np.zeros((1,1,d)) if center is None else center
        self.radius = radius
        self.p = p
        
    def __call__(self, x) :
        return np.abs(
            np.linalg.norm(x - self.center, axis=-1, ord=self.p) - self.radius
            )