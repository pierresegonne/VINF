import os

import tensorflow as tf

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Custom layers as flows """
class RadialFlow(tf.keras.layers.Layer):
    def __init__(self, d=1, init_sigma=0.01):
        super(Radial//Flow, self).__init__()

    # TODO