import os
import tensorflow as tf

from models.shared import Flows, ParametrizedGaussian

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Custom layers as flows """
class PlanarFlow(tf.keras.layers.Layer):
    def __init__(self, init_sigma=0.1):
        super(PlanarFlow, self).__init__()
        # Parameters
        self.d = None
        self.init_sigma = init_sigma

    def build(self, input_shape):
        # know the shapes of the input tensors and can do the rest of the initialization
        # parameters of the flow
        w_init = tf.random_normal_initializer(stddev=self.init_sigma)
        self.d = input_shape[-1]
        self.u = self.add_weight(
            'u',
            shape=[1, self.d],
            initializer=w_init
            )
        self.w = self.add_weight(
            'w',
            shape=[1, self.d],
            initializer=w_init
            )
        self.b = self.add_weight(
            'b',
            shape=[1],
            initializer=w_init
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
        self._normalized_u = self.u + (m(wTu) - wTu) * (self.w/wTw)
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
        return 1 - (tf.math.tanh(x)*tf.math.tanh(x))

    def psi(self, z):
        """
        z: [N,d]
        psi: [N,d]
        """
        return self.h_p(z@tf.transpose(self.w) + self.b) @ self.w

    def call(self, z_input):
        """
        z: [N,d]
        fz: [N,d], log_det_jacobian: [1]
        """
        if isinstance(z_input, tuple):
            z, log_det_jacobian = z_input[0], z_input[1]
        else:
            z, log_det_jacobian = z_input, 0

        u = self.normalized_u

        det_jacobian = 1 + self.psi(z)@tf.transpose(u)
        log_det_jacobian += tf.math.log(tf.math.abs(det_jacobian) + 1e-10)

        fz = z + self.h(z@tf.transpose(self.w) + self.b)@u

        return fz, log_det_jacobian

class PlanarFlows(Flows):
    def __init__(self, d=2, n_flows=10, shape=(1000, 2)):
        super(PlanarFlows, self).__init__(d=d, n_flows=n_flows, shape=shape)

        for i in range(1, self.n_flows + 1):
            setattr(self, "flow%i" % i, PlanarFlow())
