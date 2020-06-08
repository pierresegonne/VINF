import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from parameters import EIGHT_SCHOOL_CENTERED, SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER
from target_distributions import eight_schools_K, eight_schools_y

SAMPLES_NAME = 'eight_schools'
figsize = (8*1.403, 8)

def visualise(q, shape):
    z, mu, log_var = q(None)

    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz = tf.math.reduce_sum(normal.log_prob(z), axis=1, keepdims=True)
    qz = tf.math.exp(log_qz).numpy().flatten()

    z = z.numpy()

    # Reorder z based on probability value to make cmap look good
    q_indx_sort = qz.argsort()
    z = z[q_indx_sort]
    qz = qz[q_indx_sort]

    thetas, mus, log_tau = z[:, 0:eight_schools_K], z[:, -2], z[:, -1]

    i = 0
    thetas = thetas[:, i]

    if not EIGHT_SCHOOL_CENTERED:
        thetas = mus + thetas * np.exp(log_tau)

    # True posterior samples
    with open(f"{SAMPLES_SAVES_FOLDER}/{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        original_samples_trace = pickle.load(f)
    n_burn = 1000

    plt.figure(figsize=figsize)
    plt.axhline(y=eight_schools_y[i], color='black', linestyle='-')
    plt.scatter(np.log(original_samples_trace['tau'][n_burn:]), original_samples_trace['theta'][:, i][n_burn:],
                color='lightsteelblue', alpha=0.6, label='MCMC Samples')
    plt.scatter(log_tau, thetas, c=qz, alpha=0.8, cmap='magma', label='ADVI Samples')
    plt.annotate(rf'$y_{i+1}$', xy=(-4, eight_schools_y[0] + 1), xycoords='data')
    plt.xlabel(r'$log(\tau)$')
    plt.ylabel(rf'$\theta_{i+1}$')
    plt.ylim(-31,53)
    leg = plt.legend()
    leg.legendHandles[1].set_color('darkmagenta')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    # Histograms for latent/params
    sns.kdeplot(original_samples_trace['theta'][:, i][n_burn:], color='steelblue',
                ax=ax1, label=rf'kde($\theta_{i+1}$' + r'$)_{MCMC}$')
    count, bins, ignored = ax1.hist(thetas, 50, density=True, color='palevioletred')
    sns.kdeplot(thetas, color='darkmagenta',
                ax=ax1, label=rf'kde($\theta_{i+1}$' + r'$)_{ADVI}$')
    ax1.legend()
    ax1.set_xlabel(rf'$\theta_{i+1}$')
    ax1.set_ylabel(rf'p($\theta_{i+1}$)')
    ax1.title.set_text(rf"Distribution $\theta_{i+1}$")

    sns.kdeplot(original_samples_trace['mu'][n_burn:], color='steelblue',
                ax=ax2, label=r'kde($\mu)_{MCMC}$')
    count, bins, ignored = ax2.hist(mus, 50, density=True, color='palevioletred')
    sns.kdeplot(mus, color='darkmagenta',
                ax=ax2, label=r'kde($\mu)_{ADVI}$')
    ax2.title.set_text(r'Distribution $\mu$')
    ax2.set_xlabel(r'$\mu$')
    ax2.set_ylabel(r'p($\mu$)')
    ax2.legend()

    sns.kdeplot(original_samples_trace['tau'][n_burn:], color='steelblue',
                ax=ax3, label=r'kde($\tau)_{MCMC}$')
    count, bins, ignored = ax3.hist(tf.math.exp(log_tau), 50, density=True, color='palevioletred')
    sns.kdeplot(tf.math.exp(log_tau).numpy(), color='darkmagenta',
                ax=ax3, label=r'kde($\tau)_{ADVI}$')
    ax3.title.set_text(r'Distribution $\tau$')
    ax3.set_xlabel(r'$\tau$')
    ax3.set_ylabel(r'p($\tau$)')
    ax3.legend()

    plt.show()
