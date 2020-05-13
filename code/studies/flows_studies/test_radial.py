import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from scipy import integrate

z_ref = -0.001
z_all = np.linspace(-5, 5, 500)

q0 = st.norm(loc=0, scale=0.25)
alpha = 0.25
beta = 1.5

# alpha = 42.454113
# beta = -42.453186

# =======
def h(r, alpha):
    return 1 / (alpha + r)


def h_p(r, alpha):
    return -1 / ((alpha + r) ** 2)


def det(r, alpha, beta):
    # Note that the first term disappears as d = 1
    return np.abs(1 + beta * h(r, alpha) + beta * h_p(r, alpha) * r)


def f(z):
    r = np.linalg.norm(z - z_ref)
    return z + beta * h(r, alpha) * (z - z_ref)
# =======

q1_all = []
det_all = []
f_all = []
for z in z_all:
    r = np.linalg.norm(z - z_ref)
    q1_all.append(q0.pdf(z) * 1 / (det(r, alpha, beta)))
    det_all.append(det(r, alpha, beta))
    f_all.append(f(z))

q1 = lambda z: q0.pdf(z) / det(np.linalg.norm(z - z_ref), alpha, beta)
print("Integral of q0", integrate.quad(q0.pdf, -5, 5))
print("Integral of q1", integrate.quad(q1, -5, 5))

plt.figure()
plt.plot(z_all, q1_all, label='q(z1)')
plt.plot(z_all, det_all, label='det')
plt.plot(z_all, q0.pdf(z_all), label='q(z0)')
plt.legend()

plt.figure()
plt.plot(z_all, f_all)


plt.figure()
plt.plot(np.linspace(0, 5, 100), [det(np.linalg.norm(z - z_ref), alpha, beta) for z in np.linspace(0, 5, 100)])
plt.plot(np.linspace(0, 5, 100), [1 for _ in np.linspace(0, 5, 100)])

plt.show()
