import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import halfcauchy

N = 1000
mu = np.random.normal(loc=0, scale=5, size=N)
tau = halfcauchy.rvs(loc=0, scale=5, size=N)
theta = np.random.normal(loc=mu, scale=tau, size=N)

mask_tau = (np.log(tau) > -2) & (np.log(tau) < 3)

plt.scatter(np.log(tau[mask_tau]), theta[mask_tau], color='gray', alpha=0.6)

plt.show()