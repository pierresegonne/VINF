import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp

from distributions import *
from flows import Flows
sys.path.append("..")
from psis import psislw

DISTRIBUTION_NAME = 'figure_eight'
MODEL_FILENAME = f"temp_weights_{DISTRIBUTION_NAME}.h5"

DATA_SHAPE = (5000,2)
flows = Flows(d=2, n_flows=10, shape=DATA_SHAPE)
flows(tf.zeros(DATA_SHAPE))
flows.load_weights(MODEL_FILENAME)
z0, zk, log_det_jacobian, mu, log_var = flows(tf.zeros(DATA_SHAPE))

normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
log_q0 = normal.log_prob(z0)
log_qk = tf.math.reduce_sum(log_q0) - tf.math.reduce_sum(log_det_jacobian)
log_p = tf.math.log(pdf_2D(zk, DISTRIBUTION_NAME))

lw_out, kss = psislw(log_p - log_qk)
print(lw_out, kss)
