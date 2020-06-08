import tensorflow as tf
import tensorflow_probability as tfp
import os

from external.psis import psislw
from models.model import get_model, load_model, save_model
from parameters import *
from target_distributions import get_log_joint_pdf
from train import mean_field_elbo, flows_elbo, train

N_SAMPLES = 5000
K = 50


def q_posterior(q, model_choice, training_parameters):
    log_joint_pdf = get_log_joint_pdf(training_parameters['name'])
    if model_choice == MEAN_FIELD:
        z, mu, log_var = q(tf.zeros(training_parameters['shape']))
        normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
        log_qz = tf.math.reduce_sum(normal.log_prob(z), axis=1)
        elbo = mean_field_elbo(log_joint_pdf, z, mu, log_var)
        return z, log_qz, elbo
    elif model_choice == PLANAR_FLOWS:
        z0, zk, log_det_jacobian, mu, log_var = q(tf.zeros(training_parameters['shape']))
        normal = tfp.distributions.Normal(loc=mu, scale=tf.math.exp(0.5 * log_var))
        log_qz0 = normal.log_prob(z0)
        log_qzk = tf.math.reduce_sum(log_qz0, axis=1) - tf.math.reduce_sum(log_det_jacobian, axis=1)
        elbo = flows_elbo(log_joint_pdf, z0, zk, log_det_jacobian, mu, log_var)
        return zk, log_qzk, elbo
    elif model_choice == RADIAL_FLOWS:
        raise NotImplementedError('Radial flows not implemented yet.')


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
    demo_gmm
    energy_1
    energy_2
    energy_3
    figure_eight
    eight_schools
    """
    target = 'eight_schools'

    """ Model for Inference | Possible Choices:
    mean_field
    planar_flows
    radial_flows
    """
    model_choice = 'planar_flows'
    model_name_suffix = ''

    training_parameters = get_training_parameters(target)
    training_parameters['shape'] = (N_SAMPLES, training_parameters['shape'][1])

    q = load_model(model_choice, training_parameters, model_name_suffix=model_name_suffix)

    _elbo = np.zeros((K,))
    _khat = np.zeros((K,))
    for k in range(K):

        z, log_q, elbo = q_posterior(q, model_choice, training_parameters)
        log_p = log_joint_pdf(z, training_parameters)
        log_r = log_p - log_q

        _, khat = psislw(log_r)
        _khat[k] = khat
        _elbo[k] = elbo

    print(f'  --KHAT {_khat},\nAVG {_khat.mean()}, STD {_khat.std()}\n')
    print(f'  --ELBO {_elbo},\nAVG {_elbo.mean()}, STD {_elbo.std()}')

    # Workaround for taking the average over different training runs
    with open('diagnostic.txt', 'a') as f:
        f.write(f'[{_khat.mean()}, {_khat.std()}, {_elbo.mean()}, {_elbo.std()}],' + os.linesep)
