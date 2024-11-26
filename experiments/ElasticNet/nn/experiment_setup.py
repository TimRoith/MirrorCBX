import cbx as cbx
from cbx.dynamics import CBO
from mirrorcbx.dynamics import MirrorCBO
import torch
import torch.nn as nn
import torchvision
import cbx.utils.resampling as rsmp
from mirrorcbx.utils import ExperimentConfig, dyn_dict
from models import Perceptron
from cbx.utils.torch_utils import (
    flatten_parameters, get_param_properties, eval_losses, norm_torch, compute_consensus_torch, 
    standard_normal_torch, eval_acc, effective_sample_size,
    to_torch_dynamic
)
import numpy as np

def select_experiment(cfg):
    return MNIST_Experiment

class objective:
    def __init__(self, train_loader, N, device, model, pprop, lamda=0.1, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_iter = iter(train_loader)
        self.N = N
        self.epoch = 0
        self.device = device   
        self.loss_fct = nn.CrossEntropyLoss()
        self.model = model
        self.pprop = pprop
        self.set_batch()
        self.lamda = lamda
        
    def __call__(self, w):   
        return eval_losses(self.x, self.y, self.loss_fct, self.model, w[0,...], self.pprop) + self.reg(w)

    def reg(self, w):
        if self.lamda > 0:
            return self.lamda * (w==0).sum(axis=-1)
        else:
            return 0
    
    def set_batch(self,):
        (x,y) = next(self.data_iter, (None, None))
        if x is None:
            self.data_iter = iter(self.train_loader)
            (x,y) = next(self.data_iter)
            self.epoch += 1
        self.x = x.to(self.device)
        self.y = y.to(self.device)

def lCBO_optimize(self, sched=None):
    self.epoch = 0
    while self.f.epoch < self.epochs:
        self.step()
        if sched is not None: sched.update(self)
        self.f.set_batch()
        if self.epoch != self.f.epoch:
            self.epoch += 1
            print(30*'<>')
            print('Epoch: ' +str(self.epoch))
            if self.test_loader is not None:
                acc = eval_acc(self.f.model, self.best_particle[0, ...], self.f.pprop, self.test_loader)
                print('Accuracy: ' + str(acc.item()))
                
            # print sparsity
            sp = ((self.best_particle[0,...]==0).sum()/self.d[0])
            print('Sparsity: ' + str(sp.item()))
            print(30*'==')

def get_learning_version(dyn_cls):
    def lCBO_init(self, *args, epochs=10, test_loader=None, **kwargs):
        dyn_cls.__init__(self, *args, **kwargs)
        self.epochs = epochs
        self.test_loader = test_loader
    
    return type(dyn_cls.__name__ + str('_learning'), 
         (dyn_cls,), 
         dict(
             __init__ = lCBO_init,
             optimize=lCBO_optimize
             )
         )

class ElasticNetTorch:
    def __init__(self, delta=1.0, lamda=1.0):
        super().__init__()
        self.delta = delta
        self.lamda = lamda
    
    def __call__(self,theta):
        return (1/(2*self.delta))*torch.sum(theta**2, axis=-1) + self.lamda*torch.sum(np.abs(theta), axis=-1)
    
    def grad(self, theta):
        return (1/(self.delta))*theta + self.lamda * torch.sign(theta)
    
    def grad_conj(self, y):
        return self.delta * torch.sign(y) * torch.clamp((torch.abs(y) - self.lamda), min=0)


class MNIST_Experiment(ExperimentConfig):
    def __init__(self, conf_path):
        super().__init__(conf_path)
        self.reg_lamda = 0
        # path and data
        data_path = "../../../../datasets/" # This path directs to one level above the CBX package
        transform = torchvision.transforms.ToTensor()
        train_data = torchvision.datasets.MNIST(data_path, train=True, transform=transform, download=False)
        test_data = torchvision.datasets.MNIST(data_path, train=False, transform=transform, download=False)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=False, num_workers=0)

        # set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dyn_kwargs['device'] = self.device
        self.dyn_kwargs['test_loader'] = self.test_loader

        # init models set mirrormap and resampling
        self.init_models()
        self.set_reg()
        self.set_resampling()

    def set_reg(self,):
        dname = self.config.dyn.name
        if dname == 'MirrorCBO':
            self.dyn_kwargs['mirrormap'] = ElasticNetTorch(getattr(self.config.mirrormap, 'lamda', 1.))

    def init_models(self,):
        # model class
        model_class = Perceptron
        N = self.dyn_kwargs['N']
        models = [model_class(sizes=[784,10]) for _ in range(N)]
        self.model = models[0]
        self.pnames = [p[0] for p in self.model.named_parameters()]
        self.pprop = get_param_properties(models, pnames=self.pnames)
        self.w = flatten_parameters(models, self.pnames).to(self.device)

    def init_x(self,):
        self.init_models()
        return self.w

    def set_resampling(self,):
        if self.config.dyn.name in ['MirrorCBO']:
            resampling =  rsmp.resampling([rsmp.consensus_stagnation(patience=1)], var_name='y')
        else:
            resampling =  rsmp.resampling([rsmp.consensus_stagnation(patience=1)], var_name='x')

        self.dyn_kwargs['post_process'] = lambda dyn: resampling(dyn)
        
    def set_dyn_kwargs(self,):
        cdyn = self.config['dyn']
        self.dyn_cls = get_learning_version(to_torch_dynamic(dyn_dict[cdyn['name']]))
        self.dyn_kwargs = {k:v for k,v in cdyn.items() if k not in ['name']}

    def get_scheduler(self,):
        return effective_sample_size(maximum=1e7, name='alpha')

    def get_objective(self,):
        return objective(
            self.train_loader, 
            self.dyn_kwargs['N'], 
            self.device, 
            self.model, 
            self.pprop, 
            lamda=self.reg_lamda,
        )

    def evaluate_dynamic(self, dyn):
        if not hasattr(self, 'num_runs'): self.num_runs = 0
        self.num_runs += dyn.M

        acc = eval_acc(dyn.f.model, dyn.best_particle[0, ...], dyn.f.pprop, self.test_loader)
        sparsity = ((dyn.best_particle[0,...]==0).sum()/dyn.d[0])

        # set attributes
        for n, z in [('acc', acc), ('sparsity', sparsity)]: 
            if not hasattr(self, n): 
                setattr(self, n, z)
            else:
                zz = (self.num_runs - dyn.M) * getattr(self, n)
        
        fname = self.path + 'results/' + self.name()
        np.savetxt(fname + '_acc_sp.txt', np.array([self.acc, self.sparsity]))
        print('Sparsity: ' + str(self.sparsity), flush=True)
        print('Accuracy: ' + str(self.accuracy), flush=True)

        
