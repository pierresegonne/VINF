import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal

# Example parameters of multivariate
mu = np.array([0.5,0.5])
sig = np.array([[0.06,0.056],[0.056,0.06]])
precision = np.linalg.inv(sig)

# Create grid and multivariate normal
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
p = multivariate_normal(mu, sig)

# Factorized distribution
q1 = multivariate_normal(mu[0], 1/precision[0,0])
q2 = multivariate_normal(mu[1], 1/precision[1,1])

def factorized_pdf(pos, q1, q2):
    n1, n2, _ = pos.shape
    pos = pos.reshape(-1,2)
    pdfs = q1.pdf(pos[:,0])*q2.pdf(pos[:,1])
    return pdfs.reshape((n1,n2))

plt.contour(X,Y, p.pdf(pos), colors='green')
plt.contour(X,Y, factorized_pdf(pos, q1, q2), colors='red')

plt.show()
