import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def visualise(q, shape):
    z0, zk, ldj, mu, log_var = q(tf.zeros(shape))

    sigma2 = 0.1
    npdf = lambda x, m, v: np.exp(-(x-m)**2/(2*v))/np.sqrt(2*np.pi*v)
    prior = lambda x: npdf(x, 0, 1)
    lik = lambda x: npdf(0.5, x**2, sigma2)
    post_scaled = lambda x: prior(x)*lik(x)

    plt.figure()
    count, bins, ignored = plt.hist(z0.numpy(), 100, density=True, color='slategray', alpha=0.6)
    plt.plot(bins, post_scaled(bins), linewidth=2, color='r')
    plt.legend(['True Posterior', 'q0'])

    plt.figure()
    count, bins, ignored = plt.hist(zk.numpy(), 100, density=True, color='darkslategrey', alpha=0.6)
    plt.plot(bins, post_scaled(bins), linewidth=2, color='r')
    plt.legend(['True Posterior', 'qk'])

    plt.show()