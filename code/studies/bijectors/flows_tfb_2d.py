import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

from parameters import SAMPLES_SAVES_EXTENSION, SAMPLES_SAVES_FOLDER


def visualise(q, shape, target):
    z = q.sample(shape[0])

    try:
        with open(f"{SAMPLES_SAVES_FOLDER}/{target}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
            original_samples = np.load(f)
    except FileNotFoundError:
        original_samples = [False]

    plt.figure()
    if len(original_samples) > 0:
        plt.scatter(original_samples[:, 0], original_samples[:, 1], color='gray', alpha=0.6, label='True Posterior')
    plt.scatter(z[:, 0], z[:, 1], color='crimson', alpha=0.6, label='q')
    plt.legend()
    plt.show()
