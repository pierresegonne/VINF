import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER


def show_samples(zk, z0, mu):
    z0 = z0.numpy()
    mu = mu.numpy().flatten()
    zk = zk.numpy()

    mask_tl = (z0[:, 0] <= mu[0]) & (z0[:, 1] >= mu[1])
    mask_tr = (z0[:, 0] >= mu[0]) & (z0[:, 1] >= mu[1])
    mask_bl = (z0[:, 0] <= mu[0]) & (z0[:, 1] <= mu[1])
    mask_br = (z0[:, 0] >= mu[0]) & (z0[:, 1] <= mu[1])

    alpha = 0.5

    plt.figure()
    plt.scatter(zk[mask_tl][:, 0], zk[mask_tl][:, 1], color='red', alpha=alpha)
    plt.scatter(zk[mask_tr][:, 0], zk[mask_tr][:, 1], color='blue', alpha=alpha)
    plt.scatter(zk[mask_bl][:, 0], zk[mask_bl][:, 1], color='green', alpha=alpha)
    plt.scatter(zk[mask_br][:, 0], zk[mask_br][:, 1], color='yellow', alpha=alpha)
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')


def show_density(zk):
    # hexbin
    zk = zk.numpy()
    plt.figure()
    plt.hexbin(zk[:, 0], zk[:, 1], gridsize=(50, 50), bins='log', cmap='inferno')  # mincnt=1 to have white background
    plt.colorbar()
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')

    # kde
    plt.figure()
    ax = sns.kdeplot(zk[:, 0], zk[:, 1], cmap='magma', n_levels=10)
    ax.set_xlabel(r'$z_{1}$')
    ax.set_ylabel(r'$z_{2}$')


def show_3d_pde(z0, zk, ldj, mu, log_var):
    # 3d prob
    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(z0)

    log_qzk = (tf.math.reduce_sum(log_qz0, axis=1, keepdims=True) - ldj).numpy()
    qzk = tf.math.exp(log_qzk).numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    zk = zk.numpy()

    ax.scatter(zk[:, 0], zk[:, 1], qzk, c=qzk.flatten(), cmap='magma')
    ax.set_xlabel(r'$z_{1}$')
    ax.set_ylabel(r'$z_{2}$')
    ax.set_zlabel(r'$q_{k}(z)$')


def visualise(q, shape, target):
    z0, zk, ldj, mu, log_var = q(tf.zeros(shape))

    show_samples(z0, z0, mu)
    show_samples(zk, z0, mu)
    show_density(zk)
    show_3d_pde(z0, zk, ldj, mu, log_var)

    try:
        with open(f"{SAMPLES_SAVES_FOLDER}/{target}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
            original_samples = np.load(f)
    except FileNotFoundError:
        original_samples = []

    plt.figure()
    plt.scatter(z0[:, 0], z0[:, 1], color='crimson', alpha=0.6, label=r'$q_{0}$')
    if len(original_samples) > 0:
        plt.scatter(original_samples[:, 0], original_samples[:, 1], color='gray', alpha=0.6, label='True Posterior')
    plt.scatter(zk[:, 0], zk[:, 1], color='springgreen', alpha=0.6, label=r'$q_{k}$')
    plt.legend()

    plt.show()
