import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm


def f(z):
    return np.exp(z)


def det_jacobian_f(z):
    return np.exp(z)


def f_inv(z_p):
    return np.log(z_p)


q = lambda z: norm(loc=0, scale=1).pdf(z)
z = np.linspace(-3, 3, 500)
z_p = f(z)

z_visualisation = np.linspace(-3, 3, 10)
z_p_visualisation = f(z_visualisation)
zeros = np.zeros((10,))

q_p = lambda z_p: q(f_inv(z_p))*(1/det_jacobian_f(f_inv(z_p)))


plt.plot(z_visualisation, zeros, 'x', color='maroon')
plt.plot(z_p_visualisation, zeros, 'x', color='midnightblue')

plt.plot(z, q(z), label='Original Distribution', color='tomato')
plt.plot(z_p, q_p(z_p), label='Distribution After the Flow', color='steelblue')

plt.xlabel(r"z|z'")
plt.ylabel(r"q(z)|q'(z')")

plt.legend()
plt.show()
