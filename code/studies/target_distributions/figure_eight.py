import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal

# Example parameters of multivariate
mu1 = 1 * np.array([-1,-1])
mu2 = 1 * np.array([1,1])
sig = 0.45 * np.identity(2)

# Create grid and multivariate normal
x = np.linspace(-5,5,300)
y = np.linspace(-5,5,300)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
p1 = multivariate_normal(mu1, sig)
p2 = multivariate_normal(mu2, sig)

def mixture_pdf(p1, p2, pos, pi=0.5):
    return (1-pi)*p1.pdf(pos) + pi*p2.pdf(pos)


plt.contour(X,Y, mixture_pdf(p1, p2, pos), cmap='magma')
plt.xlabel(r'$z_{1}$')
plt.ylabel(r'$z_{2}$')
plt.show()

