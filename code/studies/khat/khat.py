import os

import tensorflow as tf
import tensorflow_probability as tfp

from external.psis import psislw
from models.model import load_model
from parameters import get_training_parameters, MEAN_FIELD, PLANAR_FLOWS, RADIAL_FLOWS
from target_distributions import get_log_joint_pdf

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def q_posterior(q, model_choice, training_parameters):
    if model_choice == MEAN_FIELD:
        z, mu, log_var = q(tf.zeros(training_parameters['shape']))
        normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
        log_qz = tf.math.reduce_sum(normal.log_prob(z), axis=1)
        return z, log_qz
    elif model_choice == PLANAR_FLOWS:
        z0, zk, log_det_jacobian, mu, log_var = q(tf.zeros(training_parameters['shape']))
        normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
        log_qz0 = normal.log_prob(z0)
        log_qzk = tf.math.reduce_sum(log_qz0, axis=1) - tf.math.reduce_sum(log_det_jacobian, axis=1)
        return zk, log_qzk
    elif model_choice == RADIAL_FLOWS:
        raise NotImplementedError('Radial flows not implemented yet.')
    elif model_choice == 'test':
        mu = tf.zeros(training_parameters['shape'][1])
        normal = tfp.distributions.Normal(loc=mu, scale=1)
        z = normal.sample(sample_shape=training_parameters['shape'][0])
        return z, tf.math.reduce_sum(normal.log_prob(z))

def log_joint_pdf(z, training_parameters):
    log_pdf = get_log_joint_pdf(training_parameters['name'])(z)
    if len(log_pdf.shape) > 1:
        return tf.math.reduce_sum(log_pdf, axis=1)
    return log_pdf


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
    model_choice = 'planar_flows'

    training_parameters = get_training_parameters(target)

    q = load_model(model_choice, training_parameters, model_name_suffix='8flows')

    z, log_q = q_posterior(q, model_choice, training_parameters)

    log_p = log_joint_pdf(z, training_parameters)

    log_r = log_p - log_q

    _, khat = psislw(log_r)

    print(f"khat: {khat}")