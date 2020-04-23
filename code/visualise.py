import os
import tensorflow as tf

from models.model import *
from parameters import *

from visualisations.mean_field_2D import visualise as visualise_mf_2D
from visualisations.mean_field_eight_schools import visualise as visualise_mf_eight_schools
from visualisations.mean_field_two_hills import visualise as visualise_mf_two_hills
from visualisations.planar_flows_2D import visualise as visualise_pf_2D
from visualisations.planar_flows_eight_schools import visualise as visualise_pf_eight_schools
from visualisations.planar_flows_two_hills import visualise as visualise_pf_two_hills


# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def visualise(q, model_choice, training_parameters):
    if model_choice == MEAN_FIELD:
        if training_parameters['name'] == 'two_hills':
            visualise_mf_two_hills(q, training_parameters['shape'])
        elif training_parameters['name'] in ['banana', 'circle', 'figure_eight']:
            visualise_mf_2D(q, training_parameters['shape'])
        elif training_parameters['name'] == 'eight_schools':
            visualise_mf_eight_schools(q, training_parameters['shape'])
    elif model_choice == PLANAR_FLOWS:
        if training_parameters['name'] == 'two_hills':
            visualise_pf_two_hills(q, training_parameters['shape'])
        elif training_parameters['name'] in ['banana', 'circle', 'figure_eight']:
            visualise_pf_2D(q, training_parameters['shape'])
        elif training_parameters['name'] == 'eight_schools':
            visualise_pf_eight_schools(q, training_parameters['shape'])
    elif model_choice == RADIAL_FLOWS:
        raise NotImplementedError


if __name__ == '__main__':

    """ Inference Target | Possible Choices:
    two_hills
    banana
    circle
    figure_eight
    eight_schools
    """
    target = 'two_hills'

    """ Model for Inference | Possible Choices:
    mean_field
    planar_flows
    """
    model_choice = 'mean_field'

    training_parameters = get_training_parameters(target)

    q = load_model(model_choice, training_parameters)

    visualise(q, model_choice, training_parameters)
