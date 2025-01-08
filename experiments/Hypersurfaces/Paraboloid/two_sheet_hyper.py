import numpy as np
from quadproj import quadrics
from quadproj.project import project

from cbx.objectives import Ackley, Rastrigin
from cbx.dynamics.polarcbo import compute_polar_consensus
from mirrorcbx.dynamics import PolarMirrorCBO
np.random.seed(4520)
#%%
class QuadricMirror:
    def __init__(self, A, b, c, off=100):
        self.Q = quadrics.Quadric(A, b, c)
        self.off = off
        
    def grad(self, x):
        return x
    
    def grad_conj(self, y):
        y = np.clip(y, a_min=-100, a_max=100)
        x = project(self.Q, y)
        idx_c = np.where(~self.Q.is_feasible(x))
        if len(idx_c[0]) > 0:
           x[idx_c] = 1#project(self.Q, x[idx_c] * self.off)
        return x 

#%%
f = Ackley(c=4*np.pi)
dim = 3
A = np.eye(dim)
A[0, 0] = 4
A[1, 1] = -2
A[2, 2] = -1
b = np.zeros(dim)
c = -1

QM = QuadricMirror(A, b, c)
x = QM.grad_conj(np.random.uniform(-1,1, size=(1,100,3)))
dyn = PolarMirrorCBO(
    f,
    mirrormap=QuadricMirror(A, b, c),
    alpha=500.,
    dt=0.05,
    sigma=0.15,
    x=x,
    track_args={'names':['x','y', 'energy']},
    max_it=400)

dyn.compute_consensus()
dyn.optimize()

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.close('all')
fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
ay = fig.add_subplot(122, projection='3d', computed_zorder=False)

m1 = 100
m2 = 100
Q = dyn.mirrormap.Q
gamma = (np.sqrt(abs(Q.c + Q.d.T @ Q.A @ Q.d + Q.b.T @ Q.d)))

g = .6
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

for axx, alpha in  [(ax, 1.), (ay, 0.4)]:
    for i in range(2):
        o,e = (i*m2//2, (i+1) * m2//2)
        surf = axx.plot_surface(
            *[X[:, o:e, i] for i in range(X.shape[-1])], 
            facecolors=plt.cm.terrain(fX[:, o:e]),
            shade=True,
            edgecolor='none',
            linewidth=0.,
            lightsource=mpl.colors.LightSource(azdeg=45, altdeg=45,),
            alpha=alpha,zorder=10,
            rstride=1, cstride=1,
            antialiased=False,
            rasterized=True
        )
    
ptcl_range = range(30,50)
num_ptcl = len(ptcl_range)
colors = iter(plt.cm.Reds(np.linspace(0.5, 1, num_ptcl)))

x = np.array(dyn.history['x'])
y = np.array(dyn.history['y'])
for j in range(num_ptcl):
    c = next(colors)
    for axx, z, zo in [(ax, x, 10), (ay, y, 0)]:
        axx.plot(*[z[:, 0, j, i] for i in range(3)], color=c, 
                zorder=zo, linewidth=2., rasterized=True)
        axx.scatter(*[z[0:1, 0, j, i] for i in range(3)], s=50, color = c, zorder=zo, marker='.', rasterized=True)
        axx.scatter(*[z[-2:-1, 0, j, i] for i in range(3)], s=50, color = 'b', zorder=zo, rasterized=True)
        
ay.plot([-5,5], [0,0],[0,0], color='k', alpha=0.5, linestyle='--')
for axx in [ax, ay]:
    axx.view_init(30, -45, 0)
    axx.set_zlim(-1,1.)
    axx.set_xlim(-1,1.)
    #axx.set_aspect('equal')
    
plt.tight_layout(pad=0., w_pad=0., h_pad=0.)
plt.savefig('results/mirrorcbo_two_sheet_vis.pdf')


