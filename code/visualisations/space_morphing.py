import matplotlib.pyplot as plt
import tensorflow as tf


def visualise_space_morphing(model):
    mesh_count = 150
    xmin = -5.
    xmax = 25.
    ymin = -10.
    ymax = 10.
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)
    pos = tf.transpose(tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])]))

    z0, zk, _, _, _ = model(pos, training=False)
    plt.figure()
    plt.plot(z0[:, 0], z0[:, 1], '-o', color='slategray', alpha=0.6, label='Original', linewidth=.5, ms=4)
    plt.plot(zk[:, 0], zk[:, 1], '-o', color='darkblue', alpha=0.9, label='Morphed', linewidth=.5, ms=4)
    plt.legend()
    plt.show()
