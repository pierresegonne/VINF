import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

sys.path.append("../")

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER, VISUALISATIONS_FOLDER

SAMPLES_NAME = 'circle'

def visualise(q, shape):
    z, mu, log_var = q(tf.zeros(shape))

    with open(f"{VISUALISATIONS_FOLDER}/{SAMPLES_SAVES_FOLDER}/{SAMPLES_NAME}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        original_samples = np.load(f)

    plt.figure()
    plt.scatter(original_samples[:,0], original_samples[:,1], color='gray', alpha=0.6)
    plt.scatter(z[:,0], z[:,1], color='crimson', alpha=0.6)
    plt.legend(['True Posterior', 'q'])
    plt.show()