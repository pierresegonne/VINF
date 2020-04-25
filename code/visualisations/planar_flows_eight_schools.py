import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
import sys

sys.path.append("../")

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER, VISUALISATIONS_FOLDER
from scipy.stats import halfcauchy

SAMPLES_NAME = 'eight_schools'

def visualise(q, shape):
    z0, zk, log_det_jacobian, mu, log_var = q(tf.zeros(shape))

    thetas0, mu0, log_tau0 = z0[:, 0], z0[:, 1], z0[:, 2]
    thetas, mu, log_tau  = zk[:, 0], zk[:, 1], zk[:, 2]

    i = 0
    # thetas0, mu0, log_tau0 = thetas0[i, :], mu0[i, :], log_tau0[i, :]
    # thetas, mu, log_tau  = thetas[i, :], mu[i, :], log_tau[i, :]

   # True posterior samples
    with open(f"{VISUALISATIONS_FOLDER}/{SAMPLES_SAVES_FOLDER}/{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        original_samples_trace = pickle.load(f)
    n_burn = 1000

    plt.figure()
    plt.scatter(log_tau0, thetas0, color='crimson', alpha=0.6)
    plt.scatter(log_tau, thetas, color='springgreen', alpha=0.6)
    plt.scatter(np.log(original_samples_trace['tau'][n_burn:]), original_samples_trace['theta'][:,0][n_burn:], color='gray', alpha=0.6)

    plt.xlabel(r'$log(\tau)$')
    plt.ylabel(r'$\theta$')

    plt.legend(['q0', 'qk', 'Posterior'])

    npdf = lambda x, m, s: np.exp(-(x-m)**2/(2*(s**2)))/np.sqrt(2*np.pi*(s**2))
    def hcpdf(x, m, s):
        pdf = np.zeros(x.shape[0])
        mask = (x >= m)
        pdf[mask] = (2/(np.pi*s))*(1/(1+((x[mask]-m)/s)**2))
        return pdf

    # Original distribution
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # Histograms for latent/params
    count, bins, ignored = ax1.hist(thetas0.numpy(), 50, density=True, color='slategrey')
    ax1.title.set_text('Distribution Theta')
    count, bins, ignored = ax2.hist(mu0.numpy(), 50, density=True, color='bisque')
    ax2.plot(bins, npdf(bins, 0, 5),
        linewidth=2, color='r')
    ax2.title.set_text('Distribution Mu')
    ax2.legend(['Prior', 'Learned Distribution'])
    count, bins, ignored = ax3.hist(tf.math.exp(log_tau0).numpy(), 50, density=True, color='skyblue')
    ax3.plot(bins, hcpdf(bins, 0, 5),
        linewidth=2, color='r')
    ax3.title.set_text('Distribution Tau')
    ax3.legend(['Prior', 'Learned Distribution'])

    # Distributions after flows
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # Histograms for latent/params
    count, bins, ignored = ax1.hist(thetas.numpy(), 50, density=True, color='darkslategrey')
    ax1.title.set_text('Distribution Theta')
    count, bins, ignored = ax2.hist(mu.numpy(), 50, density=True, color='orange')
    ax2.plot(bins, npdf(bins, 0, 5),
        linewidth=2, color='r')
    ax2.title.set_text('Distribution Mu')
    ax2.legend(['Prior', 'Learned Distribution'])
    count, bins, ignored = ax3.hist(tf.math.exp(log_tau).numpy(), 50, density=True, color='steelblue')
    ax3.plot(bins, hcpdf(bins, 0, 5),
        linewidth=2, color='r')
    ax3.title.set_text('Distribution Tau')
    ax3.legend(['Prior', 'Learned Distribution'])

    plt.show()