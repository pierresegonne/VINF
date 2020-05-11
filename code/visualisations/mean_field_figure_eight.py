import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from target_distributions import figure_eight_mu1, figure_eight_mu2, figure_eight_cov


def visualise(q, shape):
    z, mu, log_var = q(tf.zeros(shape))

    comp1 = np.random.multivariate_normal(mean=figure_eight_mu1, cov=figure_eight_cov, size=1000)
    comp2 = np.random.multivariate_normal(mean=figure_eight_mu2, cov=figure_eight_cov, size=1000)

    original_samples = np.vstack((comp1, comp2))

    plt.figure()
    plt.scatter(original_samples[:, 0], original_samples[:, 1], color='gray', alpha=0.6, label='True Posterior')
    plt.scatter(z[:, 0], z[:, 1], color='crimson', alpha=0.6, label='q')
    plt.scatter(figure_eight_mu1[0], figure_eight_mu1[1], color='darkorange', s=40, marker="x", label=r'$\mu_{1}$')
    plt.scatter(figure_eight_mu2[0], figure_eight_mu2[1], color='darkorange', s=40, marker="x", label=r'$\mu_{2}$')
    plt.legend()
    plt.show()
