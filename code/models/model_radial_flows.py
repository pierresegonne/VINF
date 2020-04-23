import tensorflow as tf

from models.shared import ParametrizedGaussian

## TODO

""" Custom layers as flows """
class RadialFlow(tf.keras.layers.Layer):
    def __init__(self, d=1, init_sigma=0.01):
        super(RadialFlow, self).__init__()

class RadialFlows(tf.keras.Model):
    def __init__(self, d=2, n_flows=10, shape=(1000, 2)):
        super(RadialFlows, self).__init__()

        # Parameters
        self.d = d
        self.n_flows = n_flows
        self.shape = shape

        # Layers
        self.parametrized_gaussian = ParametrizedGaussian()
        for i in range(1, self.n_flows + 1):
            setattr(self, "flow%i" % i, RadialFlow())