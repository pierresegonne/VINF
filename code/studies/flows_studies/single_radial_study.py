import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
For a given set of parameters that will define contractions/expansions
from a point, study the evolution of the transfer function
applied to the original distribution.
"""

# =======
# Data
resolution = 1e-2
x = np.arange(0, 2, resolution)
y = np.arange(0, 2, resolution)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

# =======
# Parameters
alpha = 1.1
beta = -0.99
#beta = -alpha + np.log(1 + np.exp(beta))
zref = np.array([[1.00001,1.00001]])

print('F Invertible: {}'.format(beta >= -alpha))

# =======
# Transfer Function
def h(r, alpha):
    return 1 / (alpha + r)

def h_p(r, alpha):
    return -1 / ((alpha + r)**2)

def detf(z, zref, alpha, beta):
    r = np.linalg.norm(z - zref, axis=1)**2 # norm 2
    return (1 + beta*h(r, alpha))*(1 + beta*h(r, alpha) + beta*h_p(r, alpha)*r)

tf = detf(pos.reshape(-1,2), zref, alpha, beta).reshape(X.shape)

# =======
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('|det f/z|')
ax.plot_surface(X,Y,tf, cmap='magma')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('-ln|det f/z|')
ax.plot_surface(X,Y,-np.log(tf), cmap='magma')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(r'|det($\frac{\partial f}{\partial z}$)$|^{-1}$')
ax.plot_surface(X,Y,np.exp(-np.log(tf)), cmap='magma')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')

plt.show()
