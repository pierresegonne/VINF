import matplotlib.pyplot as plt
import numpy as np

# Create grid and multivariate normal
x = np.linspace(-10,10,300)
y = np.linspace(-10,10,300)
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

plt.contour(X,Y, circle_pdf(pos), cmap='magma')
plt.show()