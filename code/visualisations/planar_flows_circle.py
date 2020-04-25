import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

sys.path.append("../")

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER, VISUALISATIONS_FOLDER

SAMPLES_NAME = 'circle'

def show_samples(zk, z0, mu):

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

def visualise(q, shape):
    z0, zk, ldj, mu, log_var = q(tf.zeros(shape))

    show_samples(z0, z0, mu)
    show_samples(zk, z0, mu)

    with open(f"{VISUALISATIONS_FOLDER}/{SAMPLES_SAVES_FOLDER}/{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        original_samples = np.load(f)

    plt.figure()
    plt.scatter(z0[:,0], z0[:,1], color='crimson', alpha=0.6)
    plt.scatter(original_samples[:,0], original_samples[:,1], color='gray', alpha=0.6)
    plt.scatter(zk[:,0], zk[:,1], color='springgreen', alpha=0.6)
    plt.legend(['q0', 'True Posterior', 'qk'])

    plt.show()
