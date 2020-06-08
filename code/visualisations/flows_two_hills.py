import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.integrate import quad
from target_distributions import two_hills_y, two_hills_sigma2


def visualise(q, shape):
    z0, zk, ldj, mu, log_var = q(tf.zeros(shape))
    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(z0)
    log_qzk = (tf.math.reduce_sum(log_qz0, axis=1, keepdims=True) - ldj).numpy()
    qz0 = tf.math.exp(log_qz0).numpy()
    qzk = tf.math.exp(log_qzk).numpy()

    npdf = lambda x, m, v: np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    prior = lambda x: npdf(x, 0, 1)
    lik = lambda x: npdf(two_hills_y, x ** 2, two_hills_sigma2)
    post_scaled = lambda x: prior(x) * lik(x)

    # reorder
    z0 = z0.numpy().flatten()
    qz0 = qz0.flatten()
    zk = zk.numpy().flatten()
    qzk = qzk.flatten()

    qz0 = [q for _, q in sorted(zip(z0, qz0))]
    z0 = list(sorted(z0))
    qzk = [q for _, q in sorted(zip(zk, qzk))]
    zk = list(sorted(zk))


    # integral check
    print(quad(post_scaled, -10, 10))

    plt.figure()
    count, bins, ignored = plt.hist(z0, 100, density=True, color='skyblue', alpha=0.6)
    plt.plot(bins, post_scaled(bins), linewidth=2, color='firebrick', label=r'p(z,x)')
    sns.kdeplot(z0, color='steelblue', label=r'kde($z_{k}$)')
    plt.plot(z0, qz0, linewidth=2, color='midnightblue', label='$q_{0}$($z_{0}$)')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel(r'p(z,x) | $q_{0}$(z)')

    plt.figure()
    count, bins, ignored = plt.hist(zk, 100, density=True, color='skyblue', alpha=0.6)
    plt.plot(bins, post_scaled(bins), linewidth=2, color='firebrick', label=r'p(z,x)')
    sns.kdeplot(zk, color='steelblue', label=r'kde($z_{k}$)')
    plt.plot(zk, qzk, linewidth=2, color='midnightblue', label='$q_{k}$($z_{k}$)')
    plt.legend()
    plt.xlabel('z')
    plt.ylabel(r'p(z,x) | $q_{k}$(z)')

    plt.show()
