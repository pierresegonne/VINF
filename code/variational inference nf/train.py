import tensorflow as tf
import tensorflow_probability as tfp

from distributions import *
from flows import Flows

def joint_pdf(z):
    #return pdf_1D(z, 'two_hills')
    return pdf_2D(z, 'eight_schools')

def variational_free_enery(joint_pdf, mu, log_var, z0, zk, log_det_jacobian):
    batch_size = z0.shape[0]

    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(z0)
    log_qzk = tf.math.reduce_sum(log_qz0) - tf.math.reduce_sum(log_det_jacobian)
    neg_log_likelihood = -tf.math.reduce_sum(tf.math.log(joint_pdf(zk) + 1e-10)) # * beta?
    return (log_qzk + neg_log_likelihood) / batch_size

# calls twice
@tf.function
def compute_loss(model):
    z0, zk, log_det_jacobian, mu, log_var = model(None)
    loss = variational_free_enery(joint_pdf, mu, log_var, z0, zk, log_det_jacobian)
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
    for epoch in range(1, epochs + 1):
        loss = compute_apply_gradients(flows, optimizer)

        if epoch % 100 == 0:
            print('Epoch {}, loss: {}'.format(epoch, loss))

    return flows


if __name__ == '__main__':

    # PARAMETERS
    d = 3
    DATA_SHAPE = (5000,d)
    N_FLOWS = 30
    EPOCHS = 10000
    # MISC
    TRAIN = True
    SAVE_MODEL = True
    MODEL_FILENAME = 'temp_weights_eight_schools.h5'

    # Train
    flows = Flows(d=d, n_flows=N_FLOWS, shape=DATA_SHAPE)
    flows(tf.zeros(DATA_SHAPE)) # build model
    if TRAIN:
        flows = train(flows, epochs=EPOCHS)
    if SAVE_MODEL:
        flows.save_weights(MODEL_FILENAME)
