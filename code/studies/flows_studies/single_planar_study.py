import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

"""
For a given set of parameters that will define contractions/expansions
along the hyperplane {y = 1}, study the evolution of the transfer function
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
w = np.array([[-1, 0]])
b = 1
u = np.array([[-0.95, -0.95]])

print('Hyperplane: {}x + {}y = -{}'.format(w[0,0],w[0,1],b))
print('F Invertible: {}'.format((w@u.T)[0,0] >= -1))

# =======
# Transfer Function

def h(x):
    return np.tanh(x)

def h_p(x):
    return 1 - (np.tanh(x)*np.tanh(x))

def psi(z, w, b):
        return h_p((w@(z.T)).reshape(-1,1) + b)@w

def detf(z, u, w, b):
    return np.abs(1 + (psi(z,w,b)@u.T))

tf = detf(pos.reshape(-1,2), u, w, b).reshape(X.shape)

# =======
# Plot

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