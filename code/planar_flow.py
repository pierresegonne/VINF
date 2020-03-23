import os
import tensorflow as tf

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Custom layers as flows """
class PlanarFlow(tf.keras.layers.Layer):
    def __init__(self, d=1, init_sigma=0.01):
        super(PlanarFlow, self).__init__()
        # parameters of the flow
        w_init = tf.random_normal_initializer(stddev=init_sigma, seed=100)
        self.u = self.add_weight(
            'u',
            shape=[1,d],
            initializer=w_init
            )
        self.w = self.add_weight(
            'w',
            shape=[1,d],
            initializer=w_init
            )
        self.b = self.add_weight(
            'b',
            shape=[1],
            initializer=w_init
            )

    # def build(self, input_shape):
    #     # know the shapes of the input tensors and can do the rest of the initialization
    #     pass

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

    def call(self, z):
        """
        z: [N,d]
        fz: [N,d], log_det_jacobian: [1]
        """
        if isinstance(z, tuple):
            z, log_det_jacobian = z
        else:
            z, log_det_jacobian = z, 0

        u = self.normalized_u

        det_jacobian = 1 + self.psi(z)@tf.transpose(u)
        log_det_jacobian += tf.math.log(tf.math.abs(det_jacobian) + 1e-10)

        fz = z + self.h(z@tf.transpose(self.w) + self.b)@u

        return fz, log_det_jacobian


planar_flow = PlanarFlow(d=2)
test_input = tf.random.normal(shape=(1,2), seed=100)
output = planar_flow(test_input)


print(test_input)
print('\n')
print(output)
print('\n')
print(planar_flow.w, planar_flow.u, planar_flow.normalized_u, planar_flow.b)
