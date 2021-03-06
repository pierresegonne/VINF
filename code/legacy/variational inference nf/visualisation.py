import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from flows import Flows

MODEL_FILENAME = 'temp_weights_figure_eight.h5'


def show_samples(zk, z0, mu, title):

    z0 = z0.numpy()
    mu = mu.numpy().flatten()
    zk = zk.numpy()

    mask_tl = (z0[:,0] <= mu[0]) & (z0[:,1] >= mu[1])
    mask_tr = (z0[:,0] >= mu[0]) & (z0[:,1] >= mu[1])
    mask_bl = (z0[:,0] <= mu[0]) & (z0[:,1] <= mu[1])
    mask_br = (z0[:,0] >= mu[0]) & (z0[:,1] <= mu[1])

    alpha = 0.5

    plt.figure(figsize=(8, 8))
    plt.scatter(zk[mask_tl][:, 0], zk[mask_tl][:, 1], color='red', alpha=alpha)
    plt.scatter(zk[mask_tr][:, 0], zk[mask_tr][:, 1], color='blue', alpha=alpha)
    plt.scatter(zk[mask_bl][:, 0], zk[mask_bl][:, 1], color='green', alpha=alpha)
    plt.scatter(zk[mask_br][:, 0], zk[mask_br][:, 1], color='yellow', alpha=alpha)

    # plt.xlim(-7.5, 7.5)
    # plt.ylim(-7.5, 7.5)

DATA_SHAPE = (5000,2)
flows = Flows(d=2, n_flows=10, shape=DATA_SHAPE)
flows(tf.zeros(DATA_SHAPE))
flows.load_weights(MODEL_FILENAME)
z0, zk, ldj, mu, log_var = flows(tf.zeros(DATA_SHAPE))


mu1 = 1 * np.array([-1,-1])
mu2 = 1 * np.array([1,1])
scale = 0.45 * np.array([[1,0],[0,1]])
pi = 0.5
comp1 = np.random.multivariate_normal(mean=mu1, cov=scale, size=1000)
comp2 = np.random.multivariate_normal(mean=mu2, cov=scale, size=1000)
original_samples = np.vstack((comp1, comp2))

show_samples(z0, z0, mu, "title")
show_samples(zk, z0, mu, "title")

plt.figure()
plt.scatter(z0[:,0], z0[:,1], color='crimson', alpha=0.6)
plt.scatter(original_samples[:,0], original_samples[:,1], color='gray', alpha=0.6)
plt.scatter(zk[:,0], zk[:,1], color='springgreen', alpha=0.6)
plt.scatter(mu1[0], mu1[1], color='darkorange', s=40, marker="x")
plt.scatter(mu2[0], mu2[1], color='darkorange', s=40, marker="x")
plt.legend(['q0', 'True Posterior', 'qk', 'mu1', 'mu2'])
plt.show()

plt.show()