import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from distributions import pdf_2D
from flows import Flows

def pdf(z):
    return pdf_2D(z, 'circle')

def variational_free_enery(pdf, mu, log_var, z0, zk, log_det_jacobian):
    batch_size = z0.shape[0]

    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(z0)
    log_qzk = tf.math.reduce_sum(log_qz0) - tf.math.reduce_sum(log_det_jacobian)
    neg_log_likelihood = -tf.math.reduce_sum(tf.math.log(pdf(zk) + 1e-10)) # * beta?
    return (log_qzk + neg_log_likelihood) / batch_size

# calls twice
@tf.function
def compute_loss(model):
    z0, zk, log_det_jacobian, mu, log_var = model(None)
    loss = variational_free_enery(pdf, mu, log_var, z0, zk, log_det_jacobian)
    return loss

@tf.function
def compute_apply_gradients(model, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(flows, epochs=10):
    optimizer = tf.keras.optimizers.Adam(1e-2)
    for epoch in range(epochs):
        loss = compute_apply_gradients(flows, optimizer)

        if epoch % 100 == 0:
            print('Epoch {}, loss: {}'.format(epoch, loss))

    return flows


def show_samples(sk, mu, title):

    plt.figure(figsize=(8, 8))
    plt.scatter(sk[:, 0], sk[:, 1])

    plt.xlim(-7.5, 7.5)
    plt.ylim(-7.5, 7.5)
    plt.show()


DATA_SHAPE = (5000,2)
flows = Flows(d=2, n_flows=16, shape=DATA_SHAPE)
flows(tf.zeros(DATA_SHAPE))
flows = train(flows, epochs=5000)
z0, zk, ldj, mu, log_var = flows(tf.zeros(DATA_SHAPE))
show_samples(zk, mu, "title")
