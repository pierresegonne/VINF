import os
import tensorflow as tf
import tensorflow_probability as tfp

from models.model import *
from parameters import *
from target_distributions import get_log_joint_pdf

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def mean_field_elbo(log_joint_pdf, z, mu, log_var):
    batch_size = z.shape[0]

    # Assuming that all factors are independent
    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz = tf.math.reduce_sum(normal.log_prob(z))
    neg_log_likelihood = -tf.math.reduce_sum(log_joint_pdf(z))

    return (log_qz + neg_log_likelihood) / batch_size

def planar_flows_elbo(log_joint_pdf, z0, zk, log_det_jacobian, mu, log_var):
    batch_size = z0.shape[0]

    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(z0)
    log_qzk = tf.math.reduce_sum(log_qz0) - tf.math.reduce_sum(log_det_jacobian)
    neg_log_likelihood = -tf.math.reduce_sum(log_joint_pdf(zk))

    return (log_qzk + neg_log_likelihood) / batch_size

@tf.function
def compute_loss(model, log_joint_pdf):
    if isinstance(model, MeanField):
        z, mu, log_var = model(None)
        loss = mean_field_elbo(log_joint_pdf, z, mu, log_var)
    if isinstance(model, PlanarFlows):
        z0, zk, log_det_jacobian, mu, log_var = model(None)
        loss = planar_flows_elbo(log_joint_pdf, z0, zk, log_det_jacobian, mu, log_var)
    return loss

@tf.function
def compute_apply_gradients(model, optimizer, log_joint_pdf):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, log_joint_pdf)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, training_parameters):
    optimizer = tf.keras.optimizers.Adam(1e-2)
    log_joint_pdf = get_log_joint_pdf(training_parameters['name'])
    for epoch in range(1, training_parameters['epochs']):
        loss = compute_apply_gradients(model, optimizer, log_joint_pdf)

        if epoch % 100 == 0:
            print('Epoch {}, loss: {}'.format(epoch, loss))
            # TODO record training stats

    return model


if __name__ == '__main__':

    """ Inference Target | Possible Choices:
    two_hills
    banana
    circle
    figure_eight
    eight_schools
    """
    target = 'eight_schools'

    """ Model for Inference | Possible Choices:
    mean_field
    planar_flows
    """
    model_choice = 'mean_field'

    training_parameters = get_training_parameters(target)

    q = get_model(model_choice, training_parameters)

    q = train(q, training_parameters)

    save_model(q, model_choice, target)