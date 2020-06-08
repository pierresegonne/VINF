import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import halfcauchy, norm

Y1 = 28
SIGMA = 15
MU = 0


def p(log_tau, theta):
    return norm(loc=theta, scale=SIGMA).pdf(Y1) \
           * norm(loc=MU, scale=np.exp(log_tau)).pdf(theta) \
           * norm(loc=0, scale=5).pdf(MU) \
           * halfcauchy(loc=0, scale=5).pdf(np.exp(log_tau)) \
           * np.exp(log_tau)


# Create grid and multivariate normal
x = np.linspace(-9, 6.5, 200)
y = np.linspace(-31, 53, 200)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
n1, n2 = X.shape

plt.figure(figsize = (8*1.403, 8))
plt.axhline(y=Y1, color='black', linestyle='-')
plt.annotate(r'$y_{1}$', xy=(-4, Y1 + 1), xycoords='data')
plt.axhline(y=MU, color='darkgray', linestyle='-')
plt.annotate(r'$\mu$', xy=(-4, MU + 1), xycoords='data')
plt.contour(X, Y, p(X.flatten(),Y.flatten()).reshape((n1,n2)), cmap='magma', levels=20, extend='both')
plt.xlabel(r'$\log(\tau)$')
plt.ylabel(r'$\theta_{1}$')
plt.show()