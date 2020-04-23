import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp

from distributions import *
from model import GaussianWithReparametrization
sys.path.append("..")
from psis import psislw

DISTRIBUTION_NAME = 'figure_eight'
MODEL_FILENAME = f"temp_weights_{DISTRIBUTION_NAME}.h5"

DATA_SHAPE = (5000,2)
q = GaussianWithReparametrization(d=2, shape=DATA_SHAPE)
q(tf.zeros(DATA_SHAPE))
q.load_weights(MODEL_FILENAME)
z, mu, log_var = q(tf.zeros(DATA_SHAPE))

print(mu, log_var)

x = np.linspace(-5,5,300)
y = np.linspace(-5,5,300)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

plt.contour(X,Y, pdf_2D(pos, DISTRIBUTION_NAME), cmap='magma')
plt.scatter(z[:,0], z[:,1])
plt.show()


normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
log_q = tf.math.reduce_sum(normal.log_prob(z), axis=1)
log_p = tf.math.log(pdf_2D(z, DISTRIBUTION_NAME))

print(log_q, log_p)

lw_out, kss = psislw(log_p - log_q)
print(lw_out, kss)
