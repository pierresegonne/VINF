import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tf.math
tfd = tfp.distributions
tfb = tfp.bijectors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


z = tf.linspace(-5., 5., 200)
# q0, a standard unit gaussian for example
norm = tfd.Normal(loc=0., scale=1.)
# we will apply f(z) = exp(z) to that distribution to obtain a log normal distribution
bij = tfb.Exp()
# Applying directly the bijector on q0, we get q1.
log_normal = bij(norm)

# For reference, q0, /!\ q0 and q1 don't live in the same support. i.e z0 are shifted to get z1.
plt.plot(z, norm.prob(z), label='Norm')
# Direct application of the resulting distribution
plt.plot(z, log_normal.prob(z), label='Log Norm')
# Manual operation to check that we get the same, cf ADVI.
plt.plot(z, tfm.exp(norm.log_prob(bij.inverse(z)) + bij.inverse_log_det_jacobian(z, 0)), label='Manual Log Norm')
plt.legend()
plt.show()
