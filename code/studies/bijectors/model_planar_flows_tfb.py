import os

import tensorflow as tf
import tensorflow_probability as tfp

tfm = tf.math
tfd = tfp.distributions
tfb = tfp.bijectors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PlanarFlowTFB(tfb.Bijector, tf.Module):
    """
    tf.Module to register trainable_variables
    """

    def __init__(self, d, init_sigma=0.1, **kwargs):
        super(PlanarFlowTFB, self).__init__(
            dtype=tf.float32,
            forward_min_event_ndims=0,
            inverse_min_event_ndims=0,
            **kwargs
        )
        # Shape of the flow goes from Rd to Rd
        self.d = d
        # Weights/Variables initializer
        self.init_sigma = init_sigma
        w_init = tf.random_normal_initializer(stddev=self.init_sigma)
        # Variables
        self.u = tf.Variable(
            w_init(shape=[1, self.d], dtype=tf.float32),
            dtype=tf.float32,
            name='u',
            trainable=True,
        )
        self.w = tf.Variable(
            w_init(shape=[1, self.d], dtype=tf.float32),
            dtype=tf.float32,
            name='w',
            trainable=True,
        )
        self.b = tf.Variable(
            w_init(shape=[1], dtype=tf.float32),
            dtype=tf.float32,
            name='b',
            trainable=True,
        )

    @property
    def normalized_u(self):
        """
        From Annex, this correction ensures the invertibility of
        the flow.
        """

        def m(x):
            return -1 + tf.math.log(1 + tf.math.exp(x))

        wTu = self.u @ tf.transpose(self.w)
        wTw = self.w @ tf.transpose(self.w)
        self._normalized_u = self.u + (m(wTu) - wTu) * (self.w / wTw)
        return self._normalized_u

    def h(self, x):
        """
        x: [N,] or [1]
        h: [N,] or [1]
        """
        return tf.math.tanh(x)

    def h_p(self, x):
        """
        x: [N,] or [1]
        h_p: [N,] or [1]
        """
        return 1 - (tf.math.tanh(x) * tf.math.tanh(x))

    def psi(self, z):
        """
        z: [N,d]
        psi: [N,d]
        """
        return self.h_p(z @ tf.transpose(self.w) + self.b) @ self.w

    def _f(self, z):
        u = self.normalized_u
        return z + self.h(z @ tf.transpose(self.w) + self.b) @ u

    def _forward(self, x):
        return self._f(x)

    def _inverse(self, y):
        # why?
        return self._f(y)

    def _log_det_jacobian(self, z):
        u = self.normalized_u
        det_jacobian = 1 + self.psi(z) @ tf.transpose(u)
        return tf.math.log(tf.math.abs(det_jacobian))

    def forward_log_det_jacobian(self, x, event_ndims, **kwargs):
        return -self._log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y, event_ndims, **kwargs):
        return self._log_det_jacobian(y)


class PlanarFlowsTFB(tf.keras.Model):

    def __init__(self, d=2, shape=(100, 2), n_flows=10, ):
        super(PlanarFlowsTFB, self).__init__()
        # Parameters
        self.d = d
        self.shape = shape
        self.n_flows = n_flows
        # Base distribution - MF = Multivariate normal diag
        base_distribution = tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=shape, dtype=tf.float32)
        )
        # Flows as chain of bijector
        flows = []
        for n in range(n_flows):
            flows.append(PlanarFlowTFB(self.d, name=f"flow_{n+1}"))
        bijector = tfb.Chain(list(reversed(flows)))
        self.flow = tfd.TransformedDistribution(
            distribution=base_distribution,
            bijector=bijector
        )

    def build(self, input_shape):
        """
        Needed to create the layers
        """
        eps = tf.random.normal(input_shape)
        self.flow.bijector.forward(eps)

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    def log_prob(self, *inputs):
        return self.flow.log_prob(*inputs)

    # @property
    # def _trainable_variables(self):
    #     self._trainable_variables = self.flow.trainable_variables
    #     return self._trainable_variables

    def sample(self, num):
        return self.flow.sample(num)

"""
potential to use keras model as well:
distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.MaskedAutoregressiveFlow(made),
    event_shape=[2])

x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)
"""
