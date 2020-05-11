import os

import tensorflow as tf

from models.shared import Flows

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Custom layers as flows """
class RadialFlow(tf.keras.layers.Layer):
    def __init__(self, init_sigma=0.01):
        super(RadialFlow, self).__init__()
        # Parameters
        self.d = None
        self.init_sigma = init_sigma

    def build(self, input_shape):
        # know the shapes of the input tensors and can do the rest of the initialization
        # parameters of the flow
        w_init = tf.random_normal_initializer(stddev=self.init_sigma)
        self.d = input_shape[-1]
        self.N = input_shape[0]
        self.z_ref = self.add_weight(
            'z_ref',
            shape=[1, self.d],
            initializer=w_init
            )
        self.alpha = self.add_weight(
            'alpha',
            shape=[1],
            initializer=w_init,
            #constraint=tf.keras.constraints.NonNeg(),
            )
        self.beta = self.add_weight(
            'beta',
            shape=[1],
            initializer=w_init
            )

    @property
    def normalized_beta(self):
        """
        From Annex, this correction ensures the invertibility of
        the flow.
        """
        def m(x):
            return tf.math.log(1 + tf.math.exp(x))

        self._normalized_beta = - self.alpha + m(self.beta)
        return self._normalized_beta

    def r(self, z):
        return tf.norm(z - self.z_ref)**2
        # return tf.norm(z - self.z_ref, ord='euclidean')**2

    def h(self, r):
        return 1 / (tf.nn.relu(self.alpha) + r)

    def h_p(self, r):
        return - 1 / ((tf.nn.relu(self.alpha) + r)**2)

    def call(self, z_input):
        """
        z: [N,d]
        fz: [N,d], log_det_jacobian: [1]
        """
        if isinstance(z_input, tuple):
            z, log_det_jacobian = z_input[0], z_input[1]
        else:
            z, log_det_jacobian = z_input, 0

        beta = self.normalized_beta
        print(self.beta, self.alpha, self.normalized_beta)
        r = self.r(z)

        log_det_jacobian += (self.d - 1)*tf.math.log(tf.math.abs(1 + beta*self.h(r))) \
            + tf.math.log(tf.math.abs(1 + beta*self.h(r) + beta*self.h_p(r)*r))

        fz = z + beta*self.h(r)*(z - self.z_ref)

        return fz, log_det_jacobian

class RadialFlows(Flows):
    def __init__(self, d=2, n_flows=10, shape=(1000, 2)):
        super(RadialFlows, self).__init__(d=d, n_flows=n_flows, shape=shape)

        for i in range(1, self.n_flows + 1):
            setattr(self, "flow%i" % i, RadialFlow())
