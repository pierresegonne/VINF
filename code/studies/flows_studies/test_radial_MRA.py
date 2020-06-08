import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from scipy import integrate

z_ref = -0.001
z_min, z_max = -2.5, 2.5
z_all = np.linspace(z_min, z_max, 500)

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

def fi(y):
    k = np.linalg.norm(y - z_ref)
    r = (-(alpha+beta-k) + np.sqrt((alpha+beta-k)**2 + 4*k*alpha))/(2) 
    c = (r*(1 + beta/(alpha+r)))

    return z_ref+r*(y - z_ref)/c

# =======
q1 = lambda z: q0.pdf(fi(z)) / det(np.linalg.norm(fi(z) - z_ref), alpha, beta)

q1_all = []
det_all = []
f_all = []
for z in z_all:
    r = np.linalg.norm(fi(z) - z_ref)
    q1_all.append(q1(z))
    det_all.append(det(r, alpha, beta))
    f_all.append(f(z))


print("Integral of q0", integrate.quad(q0.pdf, -np.Inf, np.Inf))
print("Integral of q1", integrate.quad(q1, -np.Inf, np.Inf))


M = 100000
z0 = q0.rvs(M)
z1 = np.array([f(zi) for zi in z0])


plt.figure()
plt.plot(z_all, q1_all, label='q(z1)')
plt.plot(z_all, det_all, label='det')
plt.plot(np.linspace(fi(z_min), fi(z_max), 500), q0.pdf(z_all), label='q(z0)') # here shouldn't be z_all actually, as z_all lives in f(z)
plt.hist(z0, 50, density=True, label='Hist q0', alpha=0.5)
plt.hist(z1, 50, density=True, label='Hist q1', alpha=0.5)
plt.legend()




fis_all = np.array([fi(zi) for zi in z_all])
plt.figure()
plt.plot(z_all, f_all, label='f')
plt.plot(z_all, fis_all, label='f inv')
plt.legend()
plt.xlabel('z')
plt.ylabel('f(z)')
plt.grid(True)



plt.figure()
plt.plot(np.linspace(0, 5, 100), [det(np.linalg.norm(fi(z) - z_ref), alpha, beta) for z in np.linspace(0, 5, 100)])
plt.plot(np.linspace(0, 5, 100), [1 for _ in np.linspace(0, 5, 100)])

plt.show()
