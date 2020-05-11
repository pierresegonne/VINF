import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../external")

from prml.rv import VariationalGaussianMixture
from serialisation import *

DATA_NAME = 'data/2D_gaussian_mixture_X.pickle'
X = read_serialised(DATA_NAME)

K = 5
vgmm = VariationalGaussianMixture(n_components=K)
vgmm.fit(X)

x0, x1 = np.meshgrid(np.linspace(-20, 20, 400), np.linspace(-20, 20, 400))
x = np.array([x0, x1]).reshape(2, -1).T

plt.scatter(X[:, 0], X[:, 1], c=vgmm.classify(X))
plt.contour(x0, x1, vgmm.pdf(x).reshape(400, 400))
plt.xlim(-20, 20, 100)
plt.ylim(-20, 20, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()