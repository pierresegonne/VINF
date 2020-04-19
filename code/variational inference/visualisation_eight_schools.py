import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import GaussianWithReparametrization
from scipy.stats import halfcauchy

MODEL_FILENAME = 'temp_weights_eight_schools.h5'

d = 3
DATA_SHAPE = (2500,3)
q = GaussianWithReparametrization(d=3, shape=DATA_SHAPE)
q(tf.zeros(DATA_SHAPE))
q.load_weights(MODEL_FILENAME)
z, mu, log_var = q(tf.zeros(DATA_SHAPE))

thetas, mu, tau  = z[:, 0], z[:, 1], z[:, 2]

# Prior
N = 5000
mu_prior = np.random.normal(loc=0, scale=5, size=N)
tau_prior = halfcauchy.rvs(loc=0, scale=5, size=N)
thetas_prior = np.random.normal(loc=mu_prior, scale=tau_prior, size=N)
mask_tau = (np.log(tau_prior) > -2) & (np.log(tau_prior) < 2.8)

plt.figure()
plt.scatter(np.log(tau_prior[mask_tau]), thetas_prior[mask_tau], color='gray', alpha=0.6)
plt.scatter(tf.math.log(tau), thetas, color='crimson', alpha=0.6)
plt.xlabel(r'$log(\tau)$')
plt.ylabel(r'$\theta$')
plt.legend(['True Posterior', 'q'])

npdf = lambda x, m, s: np.exp(-(x-m)**2/(2*(s**2)))/np.sqrt(2*np.pi*(s**2))
def hcpdf(x, m, s):
    pdf = np.zeros(x.shape[0])
    mask = (x >= m)
    pdf[mask] = (2/(np.pi*s))*(1/(1+((x[mask]-m)/s)**2))
    return pdf

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# Histograms for latent/params
count, bins, ignored = ax1.hist(thetas.numpy(), 50, density=True, color='slategrey')
ax1.title.set_text('Distribution Theta')
count, bins, ignored = ax2.hist(mu.numpy(), 50, density=True, color='bisque')
ax2.plot(bins, npdf(bins, 0, 5),
    linewidth=2, color='r')
ax2.title.set_text('Distribution Mu')
ax2.legend(['Prior', 'Learned Distribution'])
count, bins, ignored = ax3.hist(tau.numpy(), 50, density=True, color='skyblue')
ax3.plot(bins, hcpdf(bins, 0, 5),
    linewidth=2, color='r')
ax3.title.set_text('Distribution Tau')
ax3.legend(['Prior', 'Learned Distribution'])


plt.show()