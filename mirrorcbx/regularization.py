import numpy as np
from cbx.scheduler import param_update

#%%
class L1:
    def __call__(self, x):
        return np.linalg.norm(x, ord=1, axis=-1)
    
class HyperplaneDistance:
    def __init__(self, a=1, b=0):
        self.a = a
        self.norm_a = np.linalg.norm(a)
        self.b = b
    
    def __call__(self, x):
        return np.abs((self.a*x).sum(axis=-1) - self.b)/self.norm_a


reg_func_dict = {
    'L1': L1,
    'Plane': HyperplaneDistance}

def select_reg_func(func):
    if isinstance(func, dict):
        return reg_func_dict[func['name']](**{k:v for (k,v) in func.items() if not k=='name'})
    elif isinstance(func, str):
        return reg_func_dict[func]()
    else:
        return func
    
    
    
#%%
class regularize_objective:
    def __init__(self, f, reg_func, lamda = 1.):
        self.f = f
        self.reg_func = select_reg_func(reg_func)
        self.lamda = lamda
        self.lamda_broadcasted = False
        
    def __call__(self, x):
        self.broadcast_lamda(x)
        return self.f(x) + self.lamda * self.reg_func(x)
    
    def broadcast_lamda(self, x):
        if not self.lamda_broadcasted:
            M = x.shape[0]
            self.lamda = np.ones((M,1)) * self.lamda
            self.lamda_broadcasted = True
        
#%%  
def simple_param_rule_mean(dyn):
    return dyn.f.reg_func(dyn.x).mean(axis=-1)

def weighted_param_rule_mean(dyn):
    E = np.exp(-dyn.alpha * dyn.eval_f(dyn.x))
    R = dyn.f.reg_func(dyn.x)
    return (E * R).sum(axis=-1)/E.sum(axis=-1)
    
    
class regularization_paramter_scheduler(param_update):
    def __init__(
        self, name: str ='lamda', 
        theta = 1., 
        theta_max = 1e3,
        factor_theta=1.1, 
        factor_lamda=1.1,
        rule_mean = 'weighted'
    ):
        super().__init__(name=name)
        self.theta = theta
        self.theta_max = theta_max
        self.factor_theta = factor_theta
        self.factor_lamda = factor_lamda
        self.param_rule_mean = weighted_param_rule_mean if rule_mean == 'weighted' else simple_param_rule_mean
        self.params_broadcasted = False
        
    def broadcast_params(self, x):
        if not self.params_broadcasted:
            M = x.shape[0]
            self.theta = np.ones((M,)) * self.theta
            self.params_broadcasted = True
        
    def update(self, dyn):
        self.broadcast_params(dyn.x)
        m = self.param_rule_mean(dyn)
        t = (m <= 1/np.sqrt(self.theta))
        self.theta[t] *= self.factor_theta
        self.theta[~t] = np.clip(self.theta[~t]/self.factor_theta, 
                                 a_max= self.theta_max, a_min=None)
        dyn.f.lamda[~t] *= self.factor_lamda
            
        
    
