import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("..")

from scipy.stats import multivariate_normal
from serialisation import *

X = read_serialised('2D_gaussian_mixture_X.pickle')
contours = read_serialised('2D_gaussian_mixture_contours.pickle')
params = read_serialised('2D_gaussian_mixture_params.pickle')

print(params)

cov = np.array([[1.140179, 0], [0, 1.3397101]])
mu = np.array([9.195904, -3.6434307])
learned_mf = multivariate_normal(mu, cov)
pos = np.empty(contours['X'].shape + (2,))
pos[:, :, 0] = contours['X']; pos[:, :, 1] = contours['Y']

# Visualization to check interest of samples.
plt.plot(X[:,0], X[:,1], 'o', color='grey', alpha=0.5)
for contour in contours['c']:
    plt.contour(contours['X'], contours['Y'], contour, levels=6, cmap='magma')
plt.contour(contours['X'], contours['Y'], learned_mf.pdf(pos), colors='red')
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.show()