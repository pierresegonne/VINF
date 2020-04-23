"""
Gathers all parameters for the project/models
"""

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
        'name': 'figure_eight',
        'd': 2,
        'epochs': 5000,
        'n_flows': 16,
        'n_samples': 5000,
    },
    {
        'name': 'eight_schools',
        'd': 3,
        'epochs': 10000,
        'n_flows': 16,
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



## ------
## Helping functions

def get_training_parameters(target):
    assert (target in existing_distributions), f"Target distribution chosen ({target}) does not exist."

    return list(filter(lambda distribution: distribution['name'] == target, target_distributions))[0]


