import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER, VISUALISATIONS_FOLDER
from target_distributions import eight_schools_K

SAMPLES_NAME = 'eight_schools'


def visualise(q, shape):
    z0, zk, log_det_jacobian, mu, log_var = q(tf.zeros(shape))

    thetas0, mu0, log_tau0 = z0[:, 0:eight_schools_K], z0[:, -2], z0[:, -1]
    thetas, mu, log_tau = zk[:, 0:eight_schools_K], zk[:, -2], zk[:, -1]

    i = 0
    thetas0 = thetas0[:, i]
    thetas = thetas[:, i]

    # True posterior samples
    with open(f"{VISUALISATIONS_FOLDER}/{SAMPLES_SAVES_FOLDER}/{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        original_samples_trace = pickle.load(f)
    n_burn = 1000

    plt.figure()
    plt.scatter(log_tau0, thetas0, color='crimson', alpha=0.6, label=r'$q_{0}$')
    plt.scatter(np.log(original_samples_trace['tau'][n_burn:]), original_samples_trace['theta'][:, i][n_burn:],
                color='gray', alpha=0.6, label='True Posterior')
    plt.scatter(log_tau, thetas, color='springgreen', alpha=0.6, label=r'$q_{k}$')
    plt.xlabel(r'$log(\tau)$')
    plt.ylabel(rf'$\theta_{i}$')
    plt.legend()

    # Original distribution
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(r'$q_{0}$', fontsize=16)
    # Histograms for latent/params
    sns.kdeplot(original_samples_trace['theta'][:, i][n_burn:], shade=True, color="r", ax=ax1)
    count, bins, ignored = ax1.hist(thetas0.numpy(), 50, density=True, color='slategrey')
    ax1.legend(['True Posterior', 'Learned Distribution'])
    ax1.title.set_text(rf"Distribution $\theta_{i}$")

    sns.kdeplot(original_samples_trace['mu'][n_burn:], shade=True, color="r", ax=ax2)
    count, bins, ignored = ax2.hist(mu0.numpy(), 50, density=True, color='bisque')
    ax2.title.set_text(r'Distribution $\mu$')
    ax2.legend(['True Posterior', 'Learned Distribution'])

    sns.kdeplot(original_samples_trace['tau'][n_burn:], shade=True, color="r", ax=ax3)
    count, bins, ignored = ax3.hist(tf.math.exp(log_tau0).numpy(), 50, density=True, color='skyblue')
    ax3.title.set_text(r'Distribution $\tau$')
    ax3.legend(['True Posterior', 'Learned Distribution'])

    # Distributions after flows
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(r'$q_{k}$', fontsize=16)
    # Histograms for latent/params
    sns.kdeplot(original_samples_trace['theta'][:, i][n_burn:], shade=True, color="r", ax=ax1)
    count, bins, ignored = ax1.hist(thetas.numpy(), 50, density=True, color='darkslategrey')
    ax1.legend(['True Posterior', 'Learned Distribution'])
    ax1.title.set_text(rf"Distribution $\theta_{i}$")

    sns.kdeplot(original_samples_trace['mu'][n_burn:], shade=True, color="r", ax=ax2)
    count, bins, ignored = ax2.hist(mu.numpy(), 50, density=True, color='orange')
    ax2.title.set_text(r'Distribution $\mu$')
    ax2.legend(['True Posterior', 'Learned Distribution'])

    sns.kdeplot(original_samples_trace['tau'][n_burn:], shade=True, color="r", ax=ax3)
    count, bins, ignored = ax3.hist(tf.math.exp(log_tau).numpy(), 50, density=True, color='steelblue')
    ax3.title.set_text(r'Distribution $\tau$')
    ax3.legend(['True Posterior', 'Learned Distribution'])

    plt.show()
