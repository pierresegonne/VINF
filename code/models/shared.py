import tensorflow as tf

class ParametrizedGaussian(tf.keras.layers.Layer):
    def __init__(self, init_sigma=0.11):
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
    def __init__(self, d=2, n_flows=10, shape=(1000, 2)):
        super(Flows, self).__init__()

        # Parameters
        self.d = d
        self.n_flows = n_flows
        self.shape = shape

        # Layers
        self.parametrized_gaussian = ParametrizedGaussian()
        # for i in range(1, self.n_flows + 1):
        #     setattr(self, "flow%i" % i, PlanarFlow())

    def build(self, input_shape):
        """
        Needed to create the layers
        """
        eps = tf.random.normal(input_shape)
        z0 = self.parametrized_gaussian(eps)
        for i in range(1, self.n_flows + 1):
            _ = getattr(self, "flow%i" % i)(z0)

    def call(self, inputs):
        # Transform unit gaussian into parametrized q0
        eps = tf.random.normal(shape=self.shape)
        z0 = self.parametrized_gaussian(eps)

        zk, log_det_jacobian = self.flow1(z0)
        for i in range(2, self.n_flows + 1 ):
            zk, log_det_jacobian = getattr(self, "flow%i" % i)((zk, log_det_jacobian))

        return z0, zk, log_det_jacobian, self.parametrized_gaussian.mu, self.parametrized_gaussian.log_var