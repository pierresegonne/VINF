"""
Variational Inference using Mean Field approximation.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import digamma, logsumexp
from scipy.stats import multivariate_normal, wishart
from serialization import read_pickled

class VariationalGaussianMixture():

    def __init__(self, n_components=1):

        # Number of components in the model
        self.K = n_components

        # Parameters of the distributions of the parameters
        # ~~
        # parameters of the dirichlet prior
        self.alpha = None
        # means of the prior gaussian
        self.m = None
        # parameters for precision of prior gaussain
        self.beta = None
        # means of prior wishart
        self.W = None
        # dof of prior wishart
        self.v = None

    def get_params(self):
        return self.alpha, self.m, self.beta, self.W, self.v

    def init_params(self, X):
        # Dimension of data
        self.N, self.D = X.shape

        # ~~
        # Number of point per component, originally all the same
        self.Nk = (self.N/self.K) * np.ones(self.K)

        # ~~
        # Equal alpha initially, [Suggested to be mix + alpha0]
        self.alpha0 = 1/self.K
        self.alpha0 = np.ones(self.K) * self.alpha0
        self.alpha = self.alpha0 + self.Nk # 1xK
        # m, m0 as mean of data, m random centroids
        self.m0 = np.mean(X, axis=0)
        self.m = X[np.random.choice(self.N, self.K, replace=False)] # KxD
        # All beta = 1 at start
        self.beta0 = 1.
        self.beta = self.beta0 + self.Nk # 1xK
        # Simplest covariance, identity
        self.W0 = 1. * np.eye(self.D)
        self.W = np.tile(self.W0, (self.K, 1, 1)) # KxKxK
        # Number of dimensions
        self.v0 = self.D
        self.v = self.v0 + self.Nk # 1xK

    def fit(self, X, max_iters=100):
        self.init_params(X)
        for i in range(max_iters):
            params = np.hstack([p.flatten() for p in self.get_params()])
            r = self.e_step(X)
            self.m_step(X, r)
            if np.allclose(params, np.hstack([p.flatten() for p in self.get_params()])):
                break

    def e_step(self, X):
        """
        Compute the rnk
        """
        d = X[:, None, :] - self.m
        E = -0.5 * (self.D / self.beta + self.v * np.sum(
                np.einsum("kij,nkj->nki", self.W, d) * d, axis=-1))
        digam = np.sum(digamma(0.5*(self.v - np.arange(self.D).reshape(-1,1))), axis=0) # D because index change in the sum
        ln_lmbd = digam + self.D*np.log(2) + np.linalg.slogdet(self.W)[1] # 1xK [logdet returns sign on index 0]
        ln_pi = digamma(self.alpha) - digamma(np.sum(self.alpha)) # 1xK
        ln_r = ln_pi + 0.5 * ln_lmbd + E
        ln_r -= logsumexp(ln_r, axis=-1)[:, None]
        r = np.exp(ln_r) # NxK
        return r

    def m_step(self, X, r):
        self.Nk = np.sum(r, axis=0) # 1xK
        # # Statistics
        X_hat = (X.T@r / self.Nk).T
        d = X[:, None, :] - X_hat
        S = np.einsum('nki,nkj->kij', d, r[:, :, None] * d) / self.Nk[:, None, None]
        self.alpha = self.alpha0 + self.Nk # 1xK
        self.beta = self.beta0 + self.Nk # 1xK
        self.m = (self.beta0 * self.m0 + self.Nk[:, None] * X_hat) / self.beta[:, None] # KxD
        d = X_hat - self.m0
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.Nk * S.T).T
            + (self.beta0 * self.Nk * np.einsum('ki,kj->kij', d, d).T / (self.beta0 + self.Nk)).T)
        self.v = self.v0 + self.Nk # 1xK

    def classify(self, X):
        r = self.e_step(X)
        return np.argmax(r, axis=1)

    def pdf(self):
        pass


## TEST
DATA_NAME = 'data/mvg2D_X.pickle'
X = read_pickled(DATA_NAME)

vgmm = VariationalGaussianMixture(n_components=8)
vgmm.fit(X)

# Points masked by color
plt.scatter(X[:, 0], X[:, 1], c=vgmm.classify(X))
plt.xlim(-100, 100, 100)
plt.ylim(-100, 100, 100)
plt.gca().set_aspect('equal', adjustable='box')


# contour plots
# Pretty but not the right thing -> non gaussians..
x = np.linspace(-100,100,500)
y = np.linspace(-100,100,500)
X,Y = np.meshgrid(x,y)
pos = np.array([X.flatten(),Y.flatten()]).T
shape = pos.shape
pdf = vgmm.e_step(pos.reshape(-1,2)).reshape(-1,vgmm.K)
for k in range(vgmm.K):
    plt.contour(X, Y, pdf[:,k].reshape(500,500))

plt.show()