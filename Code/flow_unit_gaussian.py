import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
"""
Apply a given succession of flows, planar or radial, to a unit gaussian and observe deformation of contour plot.
"""

# =======
# Plot helpers
def plot_3d_distribution(X, Y, pdf):
    #Make a 3D plot
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, pdf, cmap='bone_r', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

def plot_contour_distribution(X,Y,pdf):
    # Contour plot
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    fig = plt.figure()
    ax = fig.gca()
    ax.contourf(X, Y, pdf, cmap='bone_r')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.grid()

# =======
# Data
resolution = 1e-2
x = np.arange(0, 2, resolution)
y = np.arange(0, 2, resolution)
X, Y = np.meshgrid(x,y)

# =======
# Original Distribution

original_mu = np.array([1,1])
original_sigma = 0.1 * np.eye(2)
rv = multivariate_normal(original_mu, original_sigma)

# ======
# Parameters of the normalizing flow
h = np.tanh
def h_p(x):
    return 1 - (np.tanh(x)*np.tanh(x))
w = np.array([0, -1])
u = np.array([0.999, 0.999])
b = 1

# ======
# Check correctness of (u,w) for f invertible
def f_invertible(u,w):
    return (w@(u.T)) >= -1

print('F invertible? {}'.format(f_invertible(u,w)))

# ======
# ~~ Manual
# K = 1 manual
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
def ln_q1(z0):
    psi = h_p(w@z0.T+b)*(w.T)
    return np.log(rv.pdf(z0)) - np.log(np.abs(1+u@psi.T))

pdf_z1 = np.zeros((pos.shape[0],pos.shape[1],1))
for i in range(pdf_z1.shape[0]):
    for j in range(pdf_z1.shape[1]):
        pdf_z1[i,j] = np.exp(ln_q1(pos[i,j]))

z0 = np.array([1, 1.5])
ln_q1(z0)

# ~~ Tensor/Matrix
def planar(z0, q0, K, u, w, b):

    def h(x):
        return np.tanh(x)

    def h_p(x):
        return 1 - (np.tanh(x)*np.tanh(x))

    def f(z, u, w, b):
        return z + h((w@(z.T)).reshape(-1,1) + b)@u[None,:]

    def fk(z, u, w, b, K):
        zk = z
        for k in range(K):
            zk = f(zk, u, w, b)
        return zk

    def psi(z, w, b):
        return h_p((w@(z.T)).reshape(-1,1) + b)@w[None,:]

    ln_qK = np.log(q0(z0))
    for k in range(1,K+1):
        ln_qK -= np.log(np.abs(1 + (psi(
            fk(z0,u[k],w[k],b[k],k-1),w[k],b[k]
            )@u[k][:,None]))).flatten()
    return ln_qK

# beware of 1 index for start
w = [0, np.array([0, -1]), np.array([-1, 0]), np.array([-1, 0]), np.array([-1, 0])]
u = [0, np.array([0.999, 0.999]), np.array([-0.9999, -0.9999]), np.array([-0.9999, -0.9999]), np.array([-0.9999, -0.9999])]
b = [0, 1, 1, 1, 1]
for i in range(1, len(b)):
    print('F invertible? {}'.format(f_invertible(u[i],w[i])))

pdf_z1 = planar(pos.reshape(-1,2), rv.pdf, 4, u, w, b)
pdf_z1 = np.exp(pdf_z1)

# Plot
if True:
    plot_contour_distribution(X,Y, pdf_z1.reshape(X.shape))
    plot_contour_distribution(X,Y, rv.pdf(pos))
    plot_3d_distribution(X,Y, pdf_z1.reshape(X.shape))
    plot_3d_distribution(X,Y, rv.pdf(pos))
plt.show()