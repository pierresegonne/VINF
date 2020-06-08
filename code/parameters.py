import numpy as np

"""
Gathers all parameters for the project/models
"""

## Number of schools in eight schools model
EIGHT_SCHOOL_K = 8
EIGHT_SCHOOL_CENTERED = False

## Target Distributions, available cases
target_distributions = [
    {
        'name': 'two_hills',
        'd': 1,
        'epochs': 10000,
        'n_flows': 4,
        'n_samples': 5000,
    },
    {
        'name': 'banana',
        'd': 2,
        'epochs': 5000,
        'n_flows': 16,
        'n_samples': 500,
    },
    {
        'name': 'circle',
        'd': 2,
        'epochs': 10000,
        'n_flows': 16,
        'n_samples': 500,
    },
    {
        'name': 'demo_gmm',
        'd': 2,
        'epochs': 5000,
        'n_flows': 16,
        'n_samples': 5000,
    },
    {
        'name': 'energy_1',
        'd': 2,
        'epochs': 15000,
        'n_flows': 32,
        'n_samples': 100,
    },
    {
        'name': 'energy_2',
        'd': 2,
        'epochs': 10000,
        'n_flows': 16,
        'n_samples': 100,
    },
    {
        'name': 'energy_3',
        'd': 2,
        'epochs': 9000,
        'n_flows': 32,
        'n_samples': 100,
    },
    {
        'name': 'energy_4',
        'd': 2,
        'epochs': 15000,
        'n_flows': 32,
        'n_samples': 100,
    },
    {
        'name': 'figure_eight',
        'd': 2,
        'epochs': 5000,
        'n_flows': 16,
        'n_samples': 5000,
    },
    {
        'name': 'eight_schools',
        'd': 2 + EIGHT_SCHOOL_K,  # 8 thetas, mu and tau
        'epochs': 15000,
        'n_flows': 64,
        'n_samples': 500,
    },
]
# list of available distribution names
existing_distributions = [distribution['name'] for distribution in target_distributions]
# update the target disributions with shape attribute
target_distributions = [{**distribution, **{'shape': (distribution['n_samples'], distribution['d'])}} for distribution
                        in target_distributions]

## Model classes
MEAN_FIELD = 'mean_field'
PLANAR_FLOWS = 'planar_flows'
RADIAL_FLOWS = 'radial_flows'
existing_models = [
    MEAN_FIELD,
    PLANAR_FLOWS,
    RADIAL_FLOWS,
]

## Saving models
MODEL_SAVES_FOLDER = 'model_saves'
MODEL_SAVE_EXTENSION = 'h5'

## Saving samples
SAMPLES_SAVES_FOLDER = 'samples'
SAMPLES_SAVES_EXTENSION = 'npy'

## Visualisations
VISUALISATIONS_FOLDER = 'visualisations'


## ------
## Helping functions

def get_training_parameters(target):
    assert (target in existing_distributions), f"Target distribution chosen ({target}) does not exist."

    return list(filter(lambda distribution: distribution['name'] == target, target_distributions))[0]


def get_training_samples(target, training_parameters):
    with open(f"{SAMPLES_SAVES_FOLDER}/{target}.{SAMPLES_SAVES_EXTENSION}", 'rb') as f:
        samples = np.load(f)
    if samples.shape[0] > training_parameters['shape'][0]:
        random_indices = np.random.choice(range(samples.shape[0]), training_parameters['shape'][0], replace=False)
        samples = samples[random_indices]
    return samples


def integral_check(model):
    mesh_count = 1000
    xmin = -5.
    xmax = 25.
    ymin = -10.
    ymax = 10.
    x = tf.linspace(xmin, xmax, mesh_count)
    y = tf.linspace(ymin, ymax, mesh_count)
    X, Y = tf.meshgrid(x, y)
    pos = tf.transpose(tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])]))

    dA = ((xmax - xmin) * (ymax - ymin)) / (mesh_count ** 2)

    z0, zk, log_det_jacobian, mu, log_var = model(pos, training=False)
    normal = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.math.exp(0.5 * log_var))
    log_qz0 = normal.log_prob(pos) # Add inverse of pos
    log_det_jacobian = tf.reshape(log_det_jacobian, (log_det_jacobian.shape[0],))
    # log_det_jacobian = tf.zeros((log_det_jacobian.shape[0],))
    log_qzk = log_qz0 - log_det_jacobian
    qzk = tf.math.exp(log_qzk)

    return tf.reduce_sum(qzk) * dA, tf.reduce_sum(normal.prob(pos)) * dA