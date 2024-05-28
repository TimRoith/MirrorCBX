import numpy as np

class ProjectionBall():
    def __init__(self, radius=3*np.sqrt(2), center=0.):
        self.radius = radius    
        self.center = center

    def constraint(self, x):
        # Calculate the sum of squares of the array elements and subtract the squared radius
        return np.sum(x**2, axis=-1) - self.radius**2

    def grad_constraint(self, x):
        # Gradient of the constraint is simply 2 * x
        return 2*x