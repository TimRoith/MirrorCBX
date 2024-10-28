import numpy as np
from quadproj import quadrics
from quadproj.project import project

from cbx.objectives import Ackley, Rastrigin
#%%

f = Ackley()
# creating random data
dim = 3
A = np.eye(dim)
A[-1,-1] = -2
#A = _A + _A.T  # make sure that A is positive definite
b = np.ones(dim)
#b[-1] = -1
c = -1.5

A[0, 0] = 4
A[1, 1] = -2
A[2, 2] = -1

b = np.array([0.5, 1, -0.25])


param = {'A': A, 'b': b, 'c': c}
Q = quadrics.Quadric(param)

x0 = np.random.uniform(-1,1, (200, dim))

s = np.linspace(-2, 2, 20)
xx = project(Q, x0)
#Q.is_feasible(x_project)
#%%
import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
m = 1000
quadric_color = 'royalblue'
flag_hyperboloid = np.any(Q.eig < 0)
m1 = 40
m2 = 20
T = np.linspace(-np.pi, np.pi, m)
x = np.zeros_like(T)
y = np.zeros_like(T)
gamma = (np.sqrt(abs(Q.c + Q.d.T @ Q.A @ Q.d + Q.b.T @ Q.d)))

g = .8
t, s1 = np.mgrid[0:2*np.pi:m1 * 1j, 0:np.pi/2-g:m2//2 * 1j]
_, s2 = np.mgrid[0:2*np.pi:m1 * 1j, np.pi/2+g:np.pi:m2//2 * 1j]
s = np.hstack((s1, s2))
t = np.hstack((t, t))
u_x = Q.axes[0] / np.cos(s)
u_y = Q.axes[1] * np.cos(t) * np.tan(s)
u_z = Q.axes[2] * np.sin(t) * np.tan(s)

U_vec = np.tile(Q.d, (m1*m2, 1)).T \
    + Q.V @ np.vstack((u_x.flatten(), u_y.flatten(), u_z.flatten())) * gamma
#    U_vec = np.tile(Q.d, (m1*m2, 1)).T
#   +  np.vstack((u_x.flatten(), u_y.flatten(), u_z.flatten()))
X = U_vec.T.reshape(m1,m2,-1)
fX = f(X)
fX = fX - fX.min()
fX = fX/fX.max()


for i in range(2):
    o,e = (i*m2//2, (i+1) * m2//2)
    surf = ax.plot_surface(*[X[:, o:e, i] for i in range(X.shape[-1])], 
                           facecolors=plt.cm.terrain(fX[:, o:e]),
                           edgecolor='none',
                           antialiased=True,
                           alpha=0.9, label=r'$\mathcal{Q}$')

#surf2 = ax.plot_surface(x2, y2, z2, color='r', alpha=0.3)
#ax.plot_wireframe(x2, y2, z2, color=quadric_color, alpha=0.7)



#ax.scatter(*[x0[..., i] for i in range(dim)], color='r')
ax.scatter(*[xx[..., i] for i in range(dim)], color='b')

d = xx - x0
# ax.quiver(*[x0[..., i] for i in range(dim)], 
#           *[d[..., i] for i in range(dim)], 
#            #scale=1., scale_units='xy', angles='xy',
#            linewidth=0.5
#            )
plt.axis('equal')

#%%
from scipy import optimize


def get_orth_vecs(b):
    l, v = np.linalg.eig(np.eye(b.shape[-1]) - np.outer(b, b))
    idx = np.where(np.isclose(l,1))[0]
    return v[:, idx].T
    
class Quadric:
    def __init__(Q, A, b, c, max_iter=500, eps=1e-2):
        Q.A = A
        Q.b = b
        Q.b_orth = get_orth_vecs(b)
        Q.c = c
        Q.max_iter = max_iter
        Q.eps = eps
        
    def project(Q, x):
        s = x.shape
        x = x.copy().reshape((-1, s[-1]))
        out, idx, idxc = Q.pre_process(x)
        x = x[idxc, :]
        mu = Q.get_mu(x)
        out[idxc] = Q.x_mu(mu, x)
        return out
    def get_mu(Q, x):
        
        #mu = np.ones(s[:-1])
        if x.shape[-1] > 0:
            mu = optimize.newton(
                Q.get_f(x), 
                np.ones(x.shape[:-1]), 
                fprime=Q.get_fdir(x), 
                args=(), 
                maxiter=Q.max_iter
            )
        return mu
    
    def pre_process(Q, x):
        nxAx = (x * (x @ Q.A)).sum(axis=-1)
        nxb  = (x * b).sum(axis=-1)
        
        AF = (np.abs(nxAx) < Q.eps) * 0
        bF = (np.abs(nxb)  < Q.eps)
        bT = (np.abs(nxb)  > Q.eps)
        
        idxA  = np.where(AF * bT)[0]
        idxAb = np.where(AF * bF)[0]
        idx   = np.where(AF)[0]
        idxc  = np.where(~AF)[0]
        
        out = np.zeros_like(x)
        if len(idxA) > 0:
            print('g')
            if np.isclose(Q.c, 0):
                ridx = np.random.randint(Q.b_orth.shape[0], size=len(idxA))
                out[idxA, :] = Q.b_orth[ridx, :]
            else:
                out[idxA, :] = -(x[idxA, :]/nxb[idxA]) * Q.c
            
            
        return out, idx, idxc
        
        
    def get_f(Q, x0):
        def f(mu):
            mu = np.atleast_1d(mu)
            z = x0 - 0.5 * mu[..., None] * Q.b
            D = np.diag(Q.A)
            inv = (1 + mu[..., None] * D)
            out = D * (z / inv)**2 + (Q.b * z)/(inv)
            return out.sum(axis=-1) - Q.c
        return f
    
    def get_fdir(Q, x0):
        def fder(mu):
            mu = np.atleast_1d(mu)
            z = x0 - 0.5 * mu[..., None] * Q.b
            D = np.diag(Q.A)
            inv = (1 + mu[..., None] * D)
            out = ( 
                - 2 * (D * z)**2       / (inv**3)
                - (D * z * Q.b)     / (inv**2)
                - (Q.b * z * D)     / (inv**2)
                - (0.5 * Q.b**2)    / inv
            )
            return out.sum(axis=-1)
        
        return fder
    
    def x_mu(Q, mu, x0):
        mu = np.atleast_1d(mu)
        z = (1/(1 + mu[...,None] * np.diag(Q.A)))
        return z * (x0 - 0.5 * mu[...,None] * Q.b)



#x0 = np.array([3., 0])
#x0 = np.array([1., 2.])
Q = Quadric(A, b, c)
mu = Q.get_mu(x0)
x = Q.x_mu(mu, x0)


#%%
import matplotlib.pyplot as plt
plt.close('all')


plt.scatter(*[x0[..., i] for i in range(dim)], color='r')
plt.scatter(*[x[..., i] for i in range(dim)])

d = x - x0

plt.plot(np.linspace(-2, 2), np.linspace(-2, 2)**2)
plt.quiver(x0[..., 0], x0[..., 1], d[..., 0], d[..., 1], 
           scale=1., scale_units='xy', angles='xy',
           width=0.001)
plt.axis('equal')

#%%
# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
# ax.scatter(*[x_project[..., i] for i in range(3)])
# idx = np.where(np.abs(x_project[:, 0]) + np.abs(x_project[:, 1]) < 0.001)
# xx = x0.reshape(-1, x0.shape[-1])[idx,:]
# ax.scatter(*[xx[..., i] for i in range(3)], color='r')
# xxx = x_project[idx,:]
# ax.scatter(*[xxx[..., i] for i in range(3)], color='g')