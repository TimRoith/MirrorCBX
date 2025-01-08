import numpy as np


def inner(x, y, A=None):
    if A is not None:
        y = (y@A.T)
    return (x * y).sum(axis=-1)

def _quasi_proj(x, Q, xi):
    b1 = inner(xi, xi, A = Q.A, )
    b2 = 2 * inner(x, xi, A = Q.A) + inner(Q.b, xi)
    b3 = inner(x, x, A = Q.A) + inner(Q.b, x) + Q.c

    det = b2**2 - 4 * b1 * b3
    vi = np.where(det > 0)[0]
    
    delta = (det[vi])**0.5
    
    beta = np.zeros(x.shape[0])
    beta[vi] = (-b2[vi] - delta)/(2 * b1[vi])
    idx = np.where(b2[vi] > 0)[0]
    beta[vi[idx]] = (-b2[vi[idx]] + delta[idx])/(2 * b1[vi[idx]])
    return x + beta[:, None] * xi, vi

def quasi_pro_retract(x, Q):
    return _quasi_proj(x, Q, Q.d - x)

def quasi_pro_grad(x, Q):
    return _quasi_proj(x, Q, 2 * (x@Q.A.T) + Q.b)

def quasi_proj(x, Q):
    xshape = x.shape
    x = x.reshape(-1, x.shape[-1])
    
    out, vi = quasi_pro_grad(x, Q)
    vii = np.ones(x.shape[0], dtype=bool)
    vii[vi] = False
    vii = np.where(vii)[0]
    out_r, vi = quasi_pro_retract(x[vii], Q)
    out[vii] = out_r
    return out.reshape(xshape)