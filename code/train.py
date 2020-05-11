import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from models.model import get_model, save_model
from models.model_mean_field import MeanField
from models.model_planar_flows import PlanarFlows
from models.model_radial_flows import RadialFlows
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


def flows_elbo(log_joint_pdf, z0, zk, log_det_jacobian, mu, log_var):
    batch_size = z0.shape[0]

    normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(z0)
    log_qzk = tf.math.reduce_sum(log_qz0) - tf.math.reduce_sum(log_det_jacobian)
    neg_log_likelihood = -tf.math.reduce_sum(log_joint_pdf(zk))

    tf.debugging.assert_all_finite(log_qz0, 'qz0 has infinite values')
    tf.debugging.assert_all_finite(log_qzk, 'qzk has infinite values')
    tf.debugging.assert_all_finite(neg_log_likelihood, 'loglik has infinite values')

    return (log_qzk + neg_log_likelihood) / batch_size


@tf.function
def compute_loss(model, log_joint_pdf):
    if isinstance(model, MeanField):
        z, mu, log_var = model(None)
        loss = mean_field_elbo(log_joint_pdf, z, mu, log_var)
    elif isinstance(model, PlanarFlows) | isinstance(model, RadialFlows):
        z0, zk, log_det_jacobian, mu, log_var = model(None)
        loss = flows_elbo(log_joint_pdf, z0, zk, log_det_jacobian, mu, log_var)
    else:
        raise ValueError(
            f"model {model} is of a not recognized model instance. Only models possible are MF, PF and RF.")
    return loss


@tf.function
def compute_apply_gradients(model, optimizer, log_joint_pdf):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, log_joint_pdf)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(model, training_parameters):
    optimizer = tf.keras.optimizers.Adam(5e-3)
    # optimizer = tf.keras.optimizers.RMSprop(1e-5, momentum=0.9)
    log_joint_pdf = get_log_joint_pdf(training_parameters['name'])

    # Early stopping
    best_loss = 1e20
    last_improvement = 0
    max_consecutive_no_improvement = 5000

    # Monitor training loss for visualisation
    loss_monitor = []
    for epoch in range(1, training_parameters['epochs']):
        loss = compute_apply_gradients(model, optimizer, log_joint_pdf)

        if loss < best_loss:
            best_loss = loss
            last_improvement = 0
            # should the best parameters be saved somehow?
        else:
            last_improvement += 1
        if last_improvement >= max_consecutive_no_improvement:
            print(f"    - STOPPED after {epoch} epochs")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss}")
            loss_monitor.append(loss)

    plt.figure()
    plt.plot(loss_monitor)

    return model


if __name__ == '__main__':

    """ Inference Target | Possible Choices:
    two_hills
    banana
    circle
    energy_1
    energy_2
    energy_3
    energy_4
    figure_eight
    eight_schools
    """
    target = 'banana'

    """ Model for Inference | Possible Choices:
    mean_field
    planar_flows
    radial_flows
    """
    model_choice = 'planar_flows'

    training_parameters = get_training_parameters(target)

    q = get_model(model_choice, training_parameters)

    q = train(q, training_parameters)

    save_model(q, model_choice, target, model_name_suffix='')

    plt.show()
