"""
Gathers all parameters for the project/models
"""

## Number of schools in eight schools model
EIGHT_SCHOOL_K = 2

## Target Distributions, available cases
target_distributions = [
    {
        'name': 'two_hills',
        'd': 1,
        'epochs': 2000,
        'n_flows': 8,
        'n_samples': 5000,
    },
    {
        'name': 'banana',
        'd': 2,
        'epochs': 5000,
        'n_flows': 16,
        'n_samples': 5000,
    },
    {
        'name': 'circle',
        'd': 2,
        'epochs': 5000,
        'n_flows': 16,
        'n_samples': 5000,
    },
    {
        'name': 'energy_1',
        'd': 2,
        'epochs': 10000,
        'n_flows': 2,
        'n_samples': 500,
    },
    {
        'name': 'energy_2',
        'd': 2,
        'epochs': 10000,
        'n_flows': 16,
        'n_samples': 5000,
    },
    {
        'name': 'energy_3',
        'd': 2,
        'epochs': 9000,
        'n_flows': 32,
        'n_samples': 5000,
    },
    {
        'name': 'energy_4',
        'd': 2,
        'epochs': 10000,
        'n_flows': 32,
        'n_samples': 5000,
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
        'd': 2 + EIGHT_SCHOOL_K, # 8 thetas, mu and tau
        'epochs': 35000,
        'n_flows': 64,
        'n_samples': 5000,
    },
]
# list of available distribution names
existing_distributions = [distribution['name'] for distribution in target_distributions]
# update the target disributions with shape attribute
target_distributions = [{**distribution, **{'shape': (distribution['n_samples'],distribution['d'])}} for distribution in target_distributions]

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


