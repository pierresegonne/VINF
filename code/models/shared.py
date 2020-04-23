import tensorflow as tf

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
