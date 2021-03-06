import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import GaussianWithReparametrization

MODEL_FILENAME = 'temp_weights_figure_eight.h5'

DATA_SHAPE = (2000,2)
q = GaussianWithReparametrization(d=2, shape=DATA_SHAPE)
q(tf.zeros(DATA_SHAPE))
q.load_weights(MODEL_FILENAME)
z, mu, log_var = q(tf.zeros(DATA_SHAPE))


mu1 = 1 * np.array([-1,-1])
mu2 = 1 * np.array([1,1])
scale = 0.45 * np.array([[1,0],[0,1]])
pi = 0.5
comp1 = np.random.multivariate_normal(mean=mu1, cov=scale, size=1000)
comp2 = np.random.multivariate_normal(mean=mu2, cov=scale, size=1000)
original_samples = np.vstack((comp1, comp2))

plt.figure()
plt.scatter(original_samples[:,0], original_samples[:,1], color='gray', alpha=0.6)
plt.scatter(z[:,0], z[:,1], color='crimson', alpha=0.6)
plt.scatter(mu1[0], mu1[1], color='darkorange', s=40, marker="x")
plt.scatter(mu2[0], mu2[1], color='darkorange', s=40, marker="x")
plt.legend(['True Posterior', 'q', 'mu1', 'mu2'])
plt.show()