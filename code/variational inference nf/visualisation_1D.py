import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from flows import Flows

MODEL_FILENAME = 'temp_weights_two_hills.h5'


DATA_SHAPE = (5000,1)
flows = Flows(d=1, n_flows=4, shape=DATA_SHAPE)
flows(tf.zeros(DATA_SHAPE))
flows.load_weights(MODEL_FILENAME)
z0, zk, ldj, mu, log_var = flows(tf.zeros(DATA_SHAPE))

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