import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import halfcauchy, multivariate_normal


N = 50

# prior
mu_prior = multivariate_normal([0], [[5]])
tau_prior = halfcauchy(loc=0, scale=5)


x = np.linspace(-10, 10, N)
pos_mask = (x >= 0)


mu_given = 10
taus = np.linspace(0.1, 8, N)
thetas = np.zeros((N,N))
for i, t in enumerate(taus):
    thetas[i,:] = np.random.normal(loc=mu_given, scale=t, size=N)

taus = taus[:,None]@np.ones((1,N))

plt.figure()
plt.plot(np.log(taus.flatten()), thetas.flatten(), 'o', color='gray', alpha=0.6)



fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, mu_prior.pdf(x))
ax2.plot(x[pos_mask], tau_prior.pdf(x[pos_mask]))








plt.show()