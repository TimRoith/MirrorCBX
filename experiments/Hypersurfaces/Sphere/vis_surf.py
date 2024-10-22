from mayavi import mlab
import numpy as np
from sphere_utils import grid_x_sph
from cbx.objectives import Ackley
#%%
d = 3
x, y = (np.load('results/mirrorcbo_' + z + '_3D.npy') for z in ('x','y'))
f = Ackley(minimum=0.4*np.ones((1,1,d)), b=0.1)
const_minimizer = 1/(d**0.5) * np.ones((d,))
#%%
X = grid_x_sph(1000, theta_max=np.pi/2)


# mesh
mesh = mlab.mesh(*[X[..., i]for i in range(3)],scalars=f(X))

# surf properties
mesh.actor.property.interpolation = 'phong'
mesh.actor.property.specular = 0.1
mesh.actor.property.specular_power = 5

# Axes
ax = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
ax.label_text_property.font_family = 'times'

# points
mlab.points3d(*list(const_minimizer), color=(1,0,0),mode='sphere',
         scale_mode='none',
         scale_factor=0.1)



lut_manager = mlab.colorbar(orientation='vertical')
lut_manager.label_text_property.font_family = 'times'
mlab.show()