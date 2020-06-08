import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tf.math
tfd = tfp.distributions
tfb = tfp.bijectors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
Verify the mechanics of dist.prob with regard to the bijector forward/inverse function
"""

norm = tfd.Normal(loc=0., scale=1.)

class Exp(tfb.Bijector):

    def __init__(self, validate_args=False, name='exp'):
      super(Exp, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

    def _forward(self, x):
      return tfm.exp(x)

    def _inverse(self, y):
      return tfm.log(y)

    def _inverse_log_det_jacobian(self, y):
      return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
      return x

zk = tf.constant(1.)

dist = tfd.TransformedDistribution(
    distribution=norm,
    bijector=Exp()
)

print(dist.log_prob(zk))
print(np.log(1/np.sqrt(2*np.pi)))

