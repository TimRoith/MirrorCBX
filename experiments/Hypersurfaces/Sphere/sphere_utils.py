import numpy as np
    
def pol2cart(phi, a=1.):
    return np.stack([ a * np.cos(phi), np.sin(phi)]).T

def sph2cart(phi):
    return np.stack([np.sin(phi[..., 1]) * np.cos(phi[..., 0]), 
                        np.sin(phi[..., 1]) * np.sin(phi[..., 0]),
                        np.cos(phi[..., 1])], axis=-1)

def grid_sph2cart(p, t):
    return sph2cart(np.stack([p, t], axis=-1))

def cart2pol(x):
    return np.arctan2(x[..., 1], x[..., 0])

def cart2sph(x):
    return np.stack([
        np.acos(x[...,2]),
        #np.sign(x[..., 1]) * np.acos(x[...,0]/np.sqrt(x[..., 0]**2+x[..., 1]**2)) + (x[..., 1] < 0)*2*np.pi,
        np.arctan2(x[...,1], x[...,0])
        ],
                       axis=-1)

def init_phi(n, axis, mode='default'):
    if axis == 2:
        return init_phi_2D(n, mode=mode)
    elif axis == 3:
        return init_phi_3D(n, mode=mode)
    else:
        raise ValueError('Only supported for 2- or 3 axisensions!')
        
def grid_phi(n, phi_max=2 * np.pi, theta_max=np.pi):
    s = int(n**0.5)
    p,t = (np.linspace(0, phi_max, s), np.linspace(0, theta_max, s))
    return np.meshgrid(p,t, indexing='ij')

def grid_x_sph(n, phi_max=2 * np.pi, theta_max=np.pi):
    p, t = grid_phi(n, phi_max=phi_max, theta_max=theta_max)
    return grid_sph2cart(p, t)

def init_phi_3D(n, mode='sunflower'):
    if mode =='grid':
        pp,tt = grid_phi(n)
        return np.stack([pp.ravel(), tt.ravel()], axis=1)
    elif mode == 'sunflower' or mode =='default':
        # this method is taken from here: 
        # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        # which offers more insights into the topic
        idx = np.arange(0, n) + 0.5
        return np.stack([np.arccos(1 - 2*idx/n), 
                            np.remainder(np.pi * (1 + 5**0.5) * idx, 2*np.pi)], axis=-1)
    else:
        raise ValueError('Unknown init mode: ' +str(mode))

def init_phi_2D(n, D = None, mode='default'):
    if mode == 'default':
        return np.linspace(-np.pi, np.pi, n+1)[:-1]
    elif mode == 'estimate':
        if D is None:
            raise ValueError('Estimate init requires the problem matrix D')
        elif abs(D[0,0] - 1) > 1e-5:
            raise ValueError('Estimate is currently implemented for d1 != 0')
        phi = np.linspace(-np.pi, np.pi, n+1)[:-1]
        xproj = np.cat([s * pol2cart(np.atan(np.tan(phi) * (D[0,0]**(3/2)))) for s in [1,-1]])
        xproj = xproj[np.where(xproj.abs()[:,1]>0.01)]
        return cart2pol(xproj)
    else:
        raise ValueError('Unknown init mode: ' +str(mode))