import tensorflow as tf

from planar_flow import PlanarFlow

def variational_free_enery():
    # TODO
    pass

class ParametrizedGaussian(tf.keras.layers.Layer):
    def __init__(self, init_sigma=0.01):
        super(ParametrizedGaussian, self).__init__()
        # Parameters
        self.d = None
        self.init_sigma = init_sigma

    def build(self, input_shape):
        self.d = input_shape[-1]
        w_init = tf.random_normal_initializer(stddev=self.init_sigma)
        self.log_var = self.add_weight(
            'log_var',
            shape=[1, self.d],
            initializer=w_init
            )
        self.mu = self.add_weight(
            'mu',
            shape=[1, self.d],
            initializer=w_init
            )

    def call(self, samples):
        std = tf.math.exp(1/2 * self.log_var)
        return self.mu + samples * std

class Flows(tf.keras.Model):
    def __init__(self, d=2, n_flows=10):
        super(Flows, self).__init__()

        # Parameters
        self.d = d
        self.n_flows = n_flows

        # Layers
        self.parametrized_gaussian = ParametrizedGaussian()
        self.flows = tf.keras.Sequential(*[PlanarFlow() for _ in range(n_flows)])

    def call(self, shape):
        # Transform unit gaussian into parametrized q0
        eps = tf.random.normal(shape=shape)
        z0 = self.parametrized_gaussian(eps)

        zk, log_det_jacobian = self.flows(z0)

        return z0, zk, log_det_jacobian,
            self.parametrized_gaussian.mu, self.parametrized_gaussian.log_var