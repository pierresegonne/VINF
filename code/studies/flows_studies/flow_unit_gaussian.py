import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal

"""
Apply a given succession of flows, planar or radial, to a unit gaussian and observe deformation of contour plot.
"""

# =======
# Plot helpers
def plot_3d_distribution(X, Y, pdf):
    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, pdf, cmap='bone_r', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

def plot_contour_distribution(X,Y,pdf):
    # Contour plot
    fig = plt.figure()
    ax = fig.gca()
    ax.contourf(X, Y, pdf, cmap='bone_r')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.grid()


# =======
# Data
resolution = 1e-2
LB = 0
UB = 2
x = np.arange(LB, UB, resolution)
y = np.arange(LB, UB, resolution)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

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

def planar_u_corrector(u,w):
    new_u = [0]
    new_u += [u[i] + (-1 + np.log(1+np.exp(w[i]@(u[i].T))) - w[i]@(u[i].T)) * w[i]/(w[i]@w[i].T) for i in range(1, len(w))]
    return new_u

# Check correctness of (alpha, beta) for f invertible [radial]
def f_invertible_radial(alpha, beta):
    return (beta >= - alpha)

def radial_beta_corrector(beta, alpha):
    new_beta = [0]
    new_beta += [-alpha[i] + np.log(1+np.exp(beta[i])) for i in range(1,len(alpha))]
    return new_beta


# ======
# Flow Computation
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
        zk_1 = fk(z0, u[k], w[k], b[k], k-1)
        ln_qK -= np.log(np.abs(1 + (psi(zk_1, w[k], b[k])@u[k][:,None]))).flatten()
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


# ======
# Parameters of the flows
# beware of 1 index for start
w = [0, np.array([0, -1]), np.array([-1, 0]), np.array([-1, 0]), np.array([-1, 0])]
u = [0, np.array([0.999, 0.999]), np.array([-0.9999, -0.9999]), np.array([-0.9999, -0.9999]), np.array([-0.9999, -0.9999])]
b = [0, 1, 1, 1, 1]
# Correction for invertibility
if True:
    u = planar_u_corrector(u, w)

zref = [0, np.array([[0.75, 0.75]]), np.array([[0.85, 0.85]]), np.array([[0.85, 0.85]])]
alpha = [0, 1.1, 0.01, 0.01]
beta = [0, -0.99, 0.5, 0.5]
# Correction for invertibility
if True:
    beta = radial_beta_corrector(beta, alpha)


# ======
# Run
n_layers = 2

PLANAR_FLOW = False
RADIAL_FLOW = not PLANAR_FLOW

if PLANAR_FLOW:
    for i in range(1, n_layers + 1):
        print('F invertible? {}'.format(f_invertible_planar(u[i],w[i])))

    pdf_zk = planar(pos.reshape(-1,2), rv.pdf, n_layers, u, w, b)
    pdf_zk = np.exp(pdf_zk)

    # For integral computation
    def qk_pdf(y,x):
        return np.exp(planar(np.array([[x,y]]), rv.pdf, n_layers, u, w, b))

if RADIAL_FLOW:
    for i in range(1, n_layers + 1):
        print('F invertible? {}'.format(f_invertible_radial(alpha[i],beta[i])))

    pdf_zk = radial(pos.reshape(-1,2), rv.pdf, n_layers, zref, alpha, beta)
    pdf_zk = np.exp(pdf_zk)

    # For integral computation
    def qk_pdf(y,x):
        return np.exp(radial(np.array([[x,y]]), rv.pdf, n_layers, zref, alpha, beta))

def q0_pdf(y,x):
    return rv.pdf(np.array([[x,y]]))

# Integrals of distributions
print('Integral of the initial distribution: {}'.format(np.trapz(np.trapz(rv.pdf(pos), pos[0, :, 0], axis=0), pos[:, 0, 1])))
print('Integral of the distribution after the flow: {}'.format(np.trapz(np.trapz(pdf_zk.reshape(X.shape), pos[0, :, 0], axis=0), pos[:, 0, 1])))
print('---')
initial_integral = dblquad(q0_pdf, LB, UB, lambda x: LB, lambda x: UB)
print('Integral of the initial distribution: {}'.format(initial_integral))
after_flow_integral = dblquad(qk_pdf, LB, UB, lambda x: LB, lambda x: UB)
print('Integral of the distribution after the flow: {}'.format(after_flow_integral))

# Plot
if True:
    plot_contour_distribution(X,Y, pdf_zk.reshape(X.shape))
    plot_contour_distribution(X,Y, rv.pdf(pos))
    plot_3d_distribution(X,Y, pdf_zk.reshape(X.shape))
    plot_3d_distribution(X,Y, rv.pdf(pos))
plt.show()