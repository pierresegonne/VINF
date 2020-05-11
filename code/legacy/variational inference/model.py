import os

import tensorflow as tf

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

class GaussianWithReparametrization(tf.keras.Model):
    def __init__(self, d=2, shape=(1000, 2)):
        super(GaussianWithReparametrization, self).__init__()

        # Parameters
        self.d = d
        self.shape = shape

        # Layers
        self.parametrized_gaussian = ParametrizedGaussian()

    def build(self, input_shape):
        """
        Needed to create the layers
        """
        eps = tf.random.normal(input_shape)
        z = self.parametrized_gaussian(eps)

    def call(self, inputs):
        # Transform unit gaussian into parametrized q0
        eps = tf.random.normal(shape=self.shape)
        z = self.parametrized_gaussian(eps)

        return z, self.parametrized_gaussian.mu, self.parametrized_gaussian.log_var