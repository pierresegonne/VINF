import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
"""
For a given set of parameters that will define contractions/expansions
along the hyperplane {y = 1}, study the evolution of the transfer function
applied to the original distribution.
"""
w = np.array([0, -1])
b = 1
u = np.array([-0.999,-0.999])

print('Hyperplane: {}x + {}y = -{}'.format(w[0],w[1],b))
print('F Invertible: {}'.format(w@u.T >= -1))

def upsi(z0):
    def h_p(x):
        return 1 - (np.tanh(x)*np.tanh(x))
    return u@(h_p(w@z0.T+b)*(w.T)).T

y = np.arange(0,2,1e-2)
z = np.zeros((y.size,2))
z[:,1] = y.reshape(-1,1)[:,0]
psi = np.zeros(y.size)
for i in range(len(psi)):
    psi[i] = upsi(z[i])

# Plot
fig = plt.figure()
ax = fig.gca()
ax.set_xlabel('Y axis')
ax.set_ylabel('u.T@Psi')
ax.grid()
ax.plot(y, psi)

fig = plt.figure()
ax = fig.gca()
ax.set_xlabel('X axis')
ax.set_ylabel('|1 + u.T@Psi|')
ax.grid()
ax.plot(y, np.abs(1 + psi))

fig = plt.figure()
ax = fig.gca()
ax.set_xlabel('Y axis')
ax.set_ylabel('- ln|1 + u.T@Psi|')
ax.grid()
ax.plot(y, - np.log(np.abs(1 + psi)))

fig = plt.figure()
ax = fig.gca()
ax.set_xlabel('Y axis')
ax.set_ylabel('exp(-ln|1 + u.T@Psi|)')
ax.grid()
ax.plot(y, np.exp(-np.log(np.abs(1 + psi))))

plt.show()