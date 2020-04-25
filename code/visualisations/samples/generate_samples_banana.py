import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")

from parameters import SAMPLES_SAVES_EXTENSION
from scipy.stats import multivariate_normal
from target_distributions import banana_mu, banana_cov

SAMPLES_NAME = 'banana'

# Create grid and multivariate normal
x = np.linspace(-2,2,200)
y = np.linspace(-2,2,200)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
p = multivariate_normal(banana_mu, banana_cov)

def banana_pdf(pos, p):
    """
    x, x**2 + y
    """
    n1, n2, _ = pos.shape
    pos = pos.reshape(-1,2)
    pos = np.vstack((pos[:,0],pos[:,0]**2 + pos[:,1])).T
    return p.pdf(pos).reshape((n1,n2))

mask = (banana_pdf(pos, p) > 0.1 - np.random.normal(loc=0, scale=0.02, size=X.shape))
pos = pos[mask]

with open(f"{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'wb') as f:
    np.save(f, pos)

plt.figure()
plt.scatter(pos[:, 0], pos[:, 1])

plt.show()
