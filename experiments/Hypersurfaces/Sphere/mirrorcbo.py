from mirrorcbx import MirrorCBO
from mirrorcbx.plotting import PlotMirrorDynamicHistory
import numpy as np
import matplotlib.pyplot as plt
from cbx.objectives import Ackley
import scienceplots
plt.style.use(['science'])
plt.rcParams['text.usetex'] = False
np.random.seed(1524023)
#%%
d = 3
x = np.random.normal(0,1, (100, 50, d))
x = x/np.linalg.norm(x, axis=-1, keepdims=True)
mode = 0

if mode == 0:
    v = np.zeros((1,1,d))#
    v[0,0,-1] = 1
    f = Ackley(minimum=v, c=3 * 2 * np.pi, b=0.2*3)
    x[..., -1] *= np.sign(x[..., -1])
    const_minimizer = v
else:
    v = 0.4 * np.ones((1,1,d))
    f = Ackley(minimum=v, c=3 * 2 * np.pi, b=0.2*3)
    const_minimizer = 1/(d**0.5) * np.ones((d,))

class proj_sphere:
    def grad(self, x):
        return x
    
    def grad_conj(self, theta):
        return theta/np.linalg.norm(theta, axis=-1, keepdims=True)


dyn = MirrorCBO(
    f,
    f_dim='3D',
    mirrormap=proj_sphere(),
    dt = 0.05,
    sigma= .25,
    noise = 'isotropic',
    alpha = 500.,
    verbosity=0,
    x=x,  max_it = 400, 
    track_args={'names':['x', 'drift', 'y', 'consensus']})


dyn.optimize()
#%%
plt.figure()
x = np.array(dyn.history['x'])
y = np.array(dyn.history['y'])
diff = np.linalg.norm(x - const_minimizer, axis=-1).mean(axis=(-2,-1))
np.save('results/mirrorcbo_diff.npy', diff)
np.save('results/mirrorcbo_x_3D.npy', np.array(dyn.history['x']))
np.save('results/mirrorcbo_y_3D.npy', np.array(dyn.history['y']))

plt.semilogy(diff)


#%%
plt.close('all')
from sphere_utils import grid_x_sph

fig = plt.figure(figsize=(15,7))
ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
ay = fig.add_subplot(122, projection='3d', computed_zorder=False)

X = grid_x_sph(100000, theta_max=np.pi/2)

f = dyn.f(X)
f = f - np.min(f)
f = f/np.max(f)

for axx, alpha in [(ax, 0.9), (ay, 0.5)]:
    axx.plot_surface(
        *[X[...,i] for i in range(3)],
        shade=False,
        edgecolor='none',
        linewidth=1.,
        #edgealpha=0.4,
        facecolors=plt.cm.terrain(f),
        alpha=alpha,zorder=1,
        rstride=1, cstride=1,
        antialiased=True,
    )

num_ptcl = 10
colors = iter(plt.cm.prism(np.linspace(0.0, 1, num_ptcl)))


for j in range(num_ptcl):
    c = next(colors)
    for axx, z, zo in [(ax, x, 10), (ay, y, 0)]:
        axx.plot(*[z[:, 0, j, i] for i in range(3)], color=c, 
                zorder=zo, linewidth=2.)
        axx.scatter(*[z[0:1, 0, j, i] for i in range(3)], s=150, color = c, zorder=zo, marker='.')
        axx.scatter(*[z[-2:-1, 0, j, i] for i in range(3)], s=50, color = 'r', zorder=zo)

ay.plot([0,0], [0,0],[0,5], color='k', alpha=0.5, linestyle='--')
for axx in [ax, ay]:
    axx.view_init(20, 30, 0)
    axx.set_zlim(0,1.)
    axx.set_aspect('equal')
    
plt.tight_layout()



