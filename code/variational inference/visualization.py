import matplotlib.pyplot as plt
import tensorflow as tf

from model import GaussianWithReparametrization

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

    plt.xlim(-7.5, 7.5)
    plt.ylim(-7.5, 7.5)

DATA_SHAPE = (5000,2)
q = GaussianWithReparametrization(d=2, shape=DATA_SHAPE)
q(tf.zeros(DATA_SHAPE))
q.load_weights(MODEL_FILENAME)
z, mu, log_var = q(tf.zeros(DATA_SHAPE))
show_samples(z, z, mu, "title")
plt.show()