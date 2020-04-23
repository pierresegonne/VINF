import tensorflow as tf

from models.shared import ParametrizedGaussian

class MeanField(tf.keras.Model):
    def __init__(self, d=2, shape=(1000, 2)):
        super(MeanField, self).__init__()

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
