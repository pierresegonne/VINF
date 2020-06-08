import os

from models.model import load_model
from parameters import *
# nf
from visualisations.flows_2d import visualise as visualise_nf_2d
from visualisations.flows_eight_schools import visualise as visualise_nf_eight_schools
from visualisations.flows_two_hills import visualise as visualise_nf_two_hills
# mf
from visualisations.mean_field_2d import visualise as visualise_mf_2d
from visualisations.mean_field_eight_schools import visualise as visualise_mf_eight_schools
from visualisations.mean_field_two_hills import visualise as visualise_mf_two_hills

# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def visualise(q, model_choice, training_parameters):
    if model_choice == MEAN_FIELD:
        if training_parameters['name'] == 'two_hills':
            visualise_mf_two_hills(q, training_parameters['shape'])
        elif training_parameters['name'] in ['banana', 'circle', 'demo_gmm', 'energy_1', 'energy_2', 'energy_3', 'energy_4', 'figure_eight']:
            visualise_mf_2d(q, training_parameters['shape'], training_parameters['name'])
        elif training_parameters['name'] == 'eight_schools':
            visualise_mf_eight_schools(q, training_parameters['shape'])
    elif (model_choice == PLANAR_FLOWS) | (model_choice == RADIAL_FLOWS):
        if training_parameters['name'] == 'two_hills':
            visualise_nf_two_hills(q, training_parameters['shape'])
        elif training_parameters['name'] in ['banana', 'circle', 'demo_gmm', 'energy_1', 'energy_2', 'energy_3', 'energy_4', 'figure_eight']:
            visualise_nf_2d(q, training_parameters['shape'], training_parameters['name'])
        elif training_parameters['name'] == 'eight_schools':
            visualise_nf_eight_schools(q, training_parameters['shape'])


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

    training_parameters = get_training_parameters(target)
    training_parameters['shape'] = (5000, training_parameters['shape'][1])

    q = load_model(model_choice, training_parameters, model_name_suffix='')

    visualise(q, model_choice, training_parameters)
