import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")

from parameters import SAMPLES_SAVES_EXTENSION
from scipy.stats import multivariate_normal

SAMPLES_NAME = 'circle'

# Create grid and multivariate normal
x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

def circle_pdf(pos):

    n1, n2, _ = pos.shape
    pos = pos.reshape(-1,2)
    z1, z2 = pos[:,0], pos[:,1]
    norm = (z1**2 + z2**2)**0.5
    exp1 = np.exp(-0.2 * ((z1 - 2) / 0.8)**2)
    exp2 = np.exp(-0.2 * ((z1 + 2) / 0.8)**2)
    u = 0.5 * ((norm - 4) / 0.4)**2 - np.log(exp1 + exp2)
    return np.exp(-u).reshape((n1,n2))

mask = (circle_pdf(pos) > 0.1 - np.random.normal(loc=0, scale=0.02, size=X.shape))
pos = pos[mask]

with open(f"{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'wb') as f:
    np.save(f, pos)

plt.figure()
plt.scatter(pos[:, 0], pos[:, 1])

plt.show()
