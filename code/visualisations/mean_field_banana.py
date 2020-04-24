import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def visualise(q, shape):
    z, mu, log_var = q(tf.zeros(shape))

    plt.figure()
    plt.scatter(z[:,0], z[:,1], color='crimson', alpha=0.6)
    plt.legend(['q'])
    plt.show()