import numpy as np
import warnings
#%%
def raise_NIE(cls_name, f_name):
    raise NotImplementedError(
        'The class ' + cls_name + 
        ' does not implement the function ' + f_name
        )
    


def Bregman_distance(F, p, q):
    return F(p) - F(q) - (F.grad(q) * (p - q)).sum(axis=-1)

def MirrorMaptoPostProcessProx(MirrorMap):
    def apply_prox(self, dyn):
       dyn.x = self.grad_conj(dyn.x)
        
    return type(MirrorMap.__name__ + str('_Prox'), 
         (MirrorMap,), 
         dict(
             __call__=apply_prox,
             )
         )


class MirrorMap:
    '''
    Abstract class for mirror maps.
    '''
    def __init__(self, ):
        pass
    
    def __call__(self, theta):
        raise raise_NIE(str(self.__class__), '__call__')
        
    def grad(self, theta):
        raise raise_NIE(str(self.__class__), 'grad')
        
    def grad_conj(self, y):
        raise raise_NIE(str(self.__class__), 'grad_conj')
        
    def hessian(self, theta):
        raise raise_NIE(str(self.__class__), 'hessian')
        
    def Bregman_distance(self, p, q):
        return Bregman_distance(self, p, q)
    
    
class ProjectionMirrorMap(MirrorMap):
    def grad(self, theta):
        return theta



class ProjectionBall(ProjectionMirrorMap):
    def __init__(self, radius=1., center=0.):
        super().__init__()
        self.radius = radius    
        self.center = center
        self.thresh = 1e-5
        
    def __call__(self, theta):
        nx = np.linalg.norm(theta, axis=-1)
        idx = np.where(nx > (self.radius + self.thresh))
        nx = 0.5*nx**2 
        nx[idx] = np.inf
        return nx
    
    def grad_conj(self, y):
        n_y = np.linalg.norm(y - self.center, axis=-1, ord=2, keepdims=True)
        return self.center + (y - self.center) / np.maximum(1, n_y/self.radius)
    
class ProjectionHyperplane(ProjectionMirrorMap):
    def __init__(self, a=1, b=0):
        super().__init__()
        self.a = a
        self.norm_a = np.linalg.norm(a, axis=-1)**2
        self.b = b
        
    def grad_conj(self, y):
        return y - ((self.a * y).sum(axis=-1, keepdims=True) - self.b)/self.norm_a * self.a


class LogBarrierBox(MirrorMap):
        
    def __call__(self, theta):
        return np.sum(np.log(1/(1-theta)) + np.log(1/(1+theta)), axis=-1)
    
    def grad(self, theta):
        return 1/(1-theta) - 1/(1+theta)
    
    def grad_conj(self, y):
        return -1/y + 1/y * np.sqrt(1 + y**2)
    
    def hessian(self, theta):
        n,m = theta.shape
        return np.expand_dims(((1/(1-theta))**2 + (1/(1+theta))**2),axis=1)*np.eye(m)
    
    
    
class L2(MirrorMap):
    
    def __call__(theta):
        return 0.5*np.sum(theta**2,axis=-1)
    
    def grad(self, theta):
        return theta
    
    def grad_conj(self, y):
        return y
    
    def hessian(self, theta):
        n,m = theta.shape
        return np.expand_dims(np.ones(theta.shape),axis=1)*np.eye(m)
   
    
   
class weighted_L2(MirrorMap):
    def __init__(self, A):
        super().__init__()
        self.A = A
    
    def __call__(self,theta):
        return 0.5*theta.T@self.A@theta
    
    def grad(self, theta):
        return np.reshape(0.5*(self.A + self.A.T)@theta[:,:,np.newaxis],theta.shape)
    
    def grad_conj(self, y):
        raise Warning('Not properly implemented')
        return np.linalg.solve(0.5*(self.A + self.A.T),y.T).T
    
    def hessian(self, theta):
        return np.expand_dims(np.ones(theta.shape),axis=1) * 0.5*(self.A.T+self.A)
    
    
    
class NonsmoothBarrier(MirrorMap):
    def __call__(self, theta):
        return np.sum(np.abs(theta)/(1-np.abs(theta)))
    
    def grad(self, theta):        
        return np.sign(theta)/(1 + np.abs(theta)**2 - 2*np.abs(theta))
    
    def grad_conj(self, y):        
        return np.sign(y) * np.maximum(1-np.sqrt(1/np.abs(y)), 0)
    
    def hessian(self, theta):
        n,m = theta.shape
        
        tmp = (-2*np.abs(theta) + 2)\
            /(-4*np.abs(theta)-4*np.abs(theta)**3+np.abs(theta)**4 + 6*np.abs(theta)**2 + 1)    
        res = np.expand_dims(tmp,axis=1)*np.eye(m)
        
        return res
    
    
    
    
class ElasticNet(MirrorMap):
    
    def __init__(self, delta=1.0, lamda=1.0):
        super().__init__()
        self.delta = delta
        self.lamda = lamda
    
    def __call__(self,theta):
        return (1/(2*self.delta))*np.sum(theta**2, axis=-1) + self.lamda*np.sum(np.abs(theta), axis=-1)
    
    def grad(self, theta):
        return (1/(self.delta))*theta + self.lamda * np.sign(theta)
    
    def grad_conj(self, y):
        return self.delta * np.sign(y) * np.maximum((np.abs(y) - self.lamda), 0)
    
    # def hessian(self, theta):
    #     n,m = theta.shape
    #     I = 1/self.delta * np.expand_dims(np.ones(theta.shape),axis=1)*np.eye(m)
        
    #     # idx,_ = np.where(np.abs(theta)<1e-12)
    #     # rho = np.zeros(theta.shape)
    #     # rho[idx] = 0e1
        
    #     # J = np.expand_dims(rho,axis=1)*np.eye(m)
    #     return I #+ J
    
    
    
    
    
mirror_dict = {
    'ElasticNet': ElasticNet,
    'None': L2, 'L2': L2,
    'ProjectionBall': ProjectionBall,
    'ProjectionHyperplane': ProjectionHyperplane,
    'LogBarrierBox': LogBarrierBox,
    'NonsmoothBarrier': NonsmoothBarrier,
    'weighted_L2': weighted_L2,
    }   


def get_mirror_map_by_name(name, **kwargs):
    if name in mirror_dict.keys():
        return mirror_dict[name](**kwargs)
    else:
        raise ValueError('Unknown mirror map ' + str(name) + '. ' + 
                         ' Please choose from ' + str(mirror_dict.keys()))
        
def get_mirror_map(mm):
    if isinstance(mm, dict):
        return get_mirror_map_by_name(
            mm['name'], 
            **{k:v for (k,v) in mm.items() if not k=='name'}
            )
    elif isinstance(mm, str) or mm is None:
        return get_mirror_map_by_name(str(mm))
    else:
        warnings.warn('MirrorMap did not fit the signature dict or str.' + 
                      'Intepreting the input as a valid mirror map.')
        return mm