import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from models.model import get_model, load_model, save_model
from models.model_mean_field import MeanField
from models.model_planar_flows import PlanarFlows
from models.model_radial_flows import RadialFlows
from parameters import *
from target_distributions import get_log_joint_pdf
from visualisations.space_morphing import visualise_space_morphing

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


def train(model, training_parameters, model_choice, target, model_name_suffix=''):
    optimizer = tf.keras.optimizers.Adam(1e-2)
    log_joint_pdf = get_log_joint_pdf(training_parameters['name'])

    # Early stopping
    best_loss = 1e20
    last_improvement = 0
    max_consecutive_no_improvement = 15000
    min_epoch_checkpoint = 1
    checkpoint_tol = 0.02
    saved_checkpoint = False

    # Monitor training loss for visualisation
    loss_monitor = []
    for epoch in range(1, training_parameters['epochs']):
        loss = compute_apply_gradients(model, optimizer, log_joint_pdf)

        if loss < best_loss:
            if ((best_loss - loss) / np.abs(best_loss) > checkpoint_tol) & (epoch > min_epoch_checkpoint) :
                print(f"    - CHECKPOINT for epoch {epoch + 1}, current best loss {loss}")
                save_model(model, model_choice, target, model_name_suffix=model_name_suffix)
                best_loss = loss
                last_improvement = 0
                saved_checkpoint = True

        else:
            last_improvement += 1
        if last_improvement >= max_consecutive_no_improvement:
            print(f"    - STOPPED after {epoch} epochs")
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, loss: {loss}")
            loss_monitor.append(loss)

    plt.figure()
    plt.plot(loss_monitor, color='slategrey')
    plt.xlabel('Epochs (x100)')
    plt.ylabel('-ELBO(q)')

    if saved_checkpoint:
        model = load_model(model_choice, training_parameters, model_name_suffix='')

    return model


if __name__ == '__main__':
    """ Inference Target | Possible Choices:
    two_hills
    banana
    circle
    demo_gmm
    energy_1
    energy_2
    energy_3
    energy_4
    figure_eight
    eight_schools
    """
    target = 'eight_schools'

    """ Model for Inference | Possible Choices:
    mean_field
    planar_flows
    radial_flows
    """
    model_choice = 'radial_flows'

    training_parameters = get_training_parameters(target)
    print(f"  - TRAINING PARAMS {training_parameters}")

    q = get_model(model_choice, training_parameters)

    start_time = time.time()
    q = train(q, training_parameters, model_choice, target, model_name_suffix='')
    end_time = time.time()
    print(f'Training time: {end_time - start_time}')

    q = load_model(model_choice, training_parameters, model_name_suffix='')

    # Opt for experiments
    # print(f"  - INTEGRAL VALUE CHECK {integral_check(q)}")
    # visualise_space_morphing(q)

    plt.show()
