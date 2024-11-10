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
    
    
def matrix_to_callable(A):
    def op(x):
        return x@A.T
    return op
    
class data_fidelity:
    def __init__(self, f, A):
        self.A = A if callable(A) else matrix_to_callable(A)
        self.f = f
        
    def __call__(self, theta):
        return 0.5 * np.linalg.norm(self.A(theta) - self.f, axis=-1)**2
    
    def grad(self, theta):
        return self.A.adjoint(self.A(theta) - self.f)