import tensorflow as tf
import tensorflow_probability as tfp

from distributions import *
from model import GaussianWithReparametrization

def joint_pdf(z):
    #return pdf_1D(z, 'two_hills')
    return pdf_2D(z, 'eight_schools')

def variational_free_enery(joint_pdf, z, mu, log_var):
    batch_size = z.shape[0]

    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz = tf.math.reduce_sum(normal.log_prob(z))
    neg_log_likelihood = -tf.math.reduce_sum(tf.math.log(joint_pdf(z) + 1e-10)) # * beta?
    return (log_qz + neg_log_likelihood) / batch_size

@tf.function
def compute_loss(model):
    z, mu, log_var = model(None)
    loss = variational_free_enery(joint_pdf, z, mu, log_var)
    return loss

@tf.function
def compute_apply_gradients(model, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, epochs=10):
    optimizer = tf.keras.optimizers.Adam(1e-2)
    for epoch in range(1, epochs + 1):
        loss = compute_apply_gradients(model, optimizer)

        if epoch % 100 == 0:
            print('Epoch {}, loss: {}'.format(epoch, loss))

    return model


if __name__ == '__main__':

    # PARAMETERS
    d = 3
    DATA_SHAPE = (3000,d)
    EPOCHS = 5000
    # MISC
    TRAIN = True
    SAVE_MODEL = True
    MODEL_FILENAME = 'temp_weights_eight_schools.h5'

    # Train
    q = GaussianWithReparametrization(d=d, shape=DATA_SHAPE)
    q(tf.zeros(DATA_SHAPE)) # build model
    if TRAIN:
        q = train(q, epochs=EPOCHS)
    if SAVE_MODEL:
        q.save_weights(MODEL_FILENAME)