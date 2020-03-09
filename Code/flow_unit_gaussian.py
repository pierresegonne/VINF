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
# Check correctness of (u,w) for f invertible [planar]
def f_invertible_planar(u,w):
    return (w@(u.T)) >= -1

# Check correctness of (alpha, beta) for f invertible [radial]
def f_invertible_radial(alpha, beta):
    return (beta >= - alpha)

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

def radial(z0, q0, K, zref, alpha, beta):

    def h(r, alpha):
        return 1 / (alpha + r)

    def h_p(r, alpha):
        return -1 / ((alpha + r)**2)

    def f(z, zref, alpha, beta):
        r = np.linalg.norm(z - zref, ord='fro')**2
        return z + beta*h(r, alpha)*(z-zref)

    def fk(z0, zref, alpha, beta, K):
        zk = z0
        for k in range(K):
            zk = f(zk, zref, alpha, beta)
        return zk

    ln_qK = np.log(q0(z0))
    for k in range(1,K+1):
        zk_1 = fk(z0, zref[k], alpha[k], beta[k], k-1)
        rk_1 = np.linalg.norm(zk_1 - zref[k], axis=1)**2 # fro or 2 ?
        ln_qK -= ((z0.shape[1]-1) * np.log(1 + beta[k]*h(rk_1, alpha[k])) +
            np.log(1 + beta[k]*h(rk_1, alpha[k]) + beta[k]*h_p(rk_1, alpha[k])*rk_1))
    return ln_qK

# beware of 1 index for start
w = [0, np.array([0, -1]), np.array([-1, 0]), np.array([-1, 0]), np.array([-1, 0])]
u = [0, np.array([0.999, 0.999]), np.array([-0.9999, -0.9999]), np.array([-0.9999, -0.9999]), np.array([-0.9999, -0.9999])]
b = [0, 1, 1, 1, 1]

zref = [0, np.array([[1.15, 1]]), np.array([[1.15, 1]]), np.array([[1.15, 1]])]
alpha = [0, 1, 0.1, 0.1]
beta = [0, -0.05, -0.001, -0.001]

n_layers = 1

PLANAR_FLOW = False
RADIAL_FLOW = not PLANAR_FLOW

if PLANAR_FLOW:
    for i in range(1, n_layers + 1):
        print('F invertible? {}'.format(f_invertible_planar(u[i],w[i])))

    pdf_z1 = planar(pos.reshape(-1,2), rv.pdf, n_layers, u, w, b)
    pdf_z1 = np.exp(pdf_z1)
if RADIAL_FLOW:
    for i in range(1, n_layers + 1):
        print('F invertible? {}'.format(f_invertible_radial(alpha[i],beta[i])))

    pdf_z1 = radial(pos.reshape(-1,2), rv.pdf, n_layers, zref, alpha, beta)
    pdf_z1 = np.exp(pdf_z1)

# Integrals of distributions
print('Integral of the initial distribution: {}'.format(np.trapz(np.trapz(rv.pdf(pos), pos[0, :, 0], axis=0), pos[:, 0, 1])))
print('Integral of the distribution after the flow: {}'.format(np.trapz(np.trapz(pdf_z1.reshape(X.shape), pos[0, :, 0], axis=0), pos[:, 0, 1])))

# Plot
if True:
    plot_contour_distribution(X,Y, pdf_z1.reshape(X.shape))
    plot_contour_distribution(X,Y, rv.pdf(pos))
    plot_3d_distribution(X,Y, pdf_z1.reshape(X.shape))
    plot_3d_distribution(X,Y, rv.pdf(pos))
plt.show()