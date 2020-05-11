import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER, VISUALISATIONS_FOLDER
from target_distributions import eight_schools_K

SAMPLES_NAME = 'eight_schools'


def visualise(q, shape):
    z, mu, log_var = q(tf.zeros(shape))

    thetas, mu, log_tau = z[:, 0:eight_schools_K], z[:, -2], z[:, -1]

    i = 0
    thetas = thetas[:, i]

    # True posterior samples
    with open(f"{VISUALISATIONS_FOLDER}/{SAMPLES_SAVES_FOLDER}/{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        original_samples_trace = pickle.load(f)
    n_burn = 1000

    plt.figure()
    plt.scatter(np.log(original_samples_trace['tau'][n_burn:]), original_samples_trace['theta'][:, i][n_burn:],
                color='gray', alpha=0.6, label='True Posterior')
    plt.scatter(log_tau, thetas, color='crimson', alpha=0.6, label='q')
    plt.xlabel(r'$log(\tau)$')
    plt.ylabel(rf'$\theta_{i}$')
    plt.legend()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # Histograms for latent/params
    sns.kdeplot(original_samples_trace['theta'][:, i][n_burn:], shade=True, color="r", ax=ax1)
    count, bins, ignored = ax1.hist(thetas.numpy(), 50, density=True, color='slategrey')
    ax1.legend(['True Posterior', 'Learned Distribution'])
    ax1.title.set_text(rf"Distribution $\theta_{i}$")

    sns.kdeplot(original_samples_trace['mu'][n_burn:], shade=True, color="r", ax=ax2)
    count, bins, ignored = ax2.hist(mu.numpy(), 50, density=True, color='bisque')
    ax2.title.set_text(r'Distribution $\mu$')
    ax2.legend(['True Posterior', 'Learned Distribution'])

    sns.kdeplot(original_samples_trace['tau'][n_burn:], shade=True, color="r", ax=ax3)
    count, bins, ignored = ax3.hist(tf.math.exp(log_tau).numpy(), 50, density=True, color='skyblue')
    ax3.title.set_text(r'Distribution $\tau$')
    ax3.legend(['True Posterior', 'Learned Distribution'])

    plt.show()
