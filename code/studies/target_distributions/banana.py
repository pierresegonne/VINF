import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal

# Example parameters of multivariate
banana_mu2 = 0
banana_std2 = 4
banana_std1 = 1

# Create grid and multivariate normal
x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# Not efficient but simple implementation purely for visualisation
def banana_pdf(pos):
    """
    N(x2|0,4)N(x1|(1/4)*x2**2,1)
    """
    p2 = multivariate_normal(banana_mu2, banana_std2)

    def banana_pdf_direct(x1, x2):
        p1pdf = multivariate_normal(x2 ** 2 / 4, banana_std1).pdf(x1)
        return p1pdf * p2.pdf(x2)

    n1, n2, _ = pos.shape
    pos = pos.reshape(-1, 2)
    x1, x2 = pos[:, 0], pos[:, 1]
    pdfs = np.zeros(x1.shape)
    for i in range(x1.shape[0]):
        pdfs[i] = banana_pdf_direct(x1[i], x2[i])
    return pdfs.reshape((n1, n2))


plt.figure()
plt.contour(X, Y, banana_pdf(pos), cmap='magma')
plt.show()
