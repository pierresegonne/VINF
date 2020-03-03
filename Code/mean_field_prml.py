import matplotlib.pyplot as plt
import numpy as np

from prml.rv import VariationalGaussianMixture
from serialization import read_pickled

DATA_NAME = 'data/mvg2D_X.pickle'
X = read_pickled(DATA_NAME)

vgmm = VariationalGaussianMixture(n_components=2)
vgmm.fit(X)

x0, x1 = np.meshgrid(np.linspace(-20, 20, 400), np.linspace(-20, 20, 400))
x = np.array([x0, x1]).reshape(2, -1).T

plt.scatter(X[:, 0], X[:, 1], c=vgmm.classify(X))
# plt.contour(x0, x1, vgmm.pdf(x).reshape(400, 400))
plt.xlim(-20, 20, 100)
plt.ylim(-20, 20, 100)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()