import os
import tensorflow as tf

from models.model import *
from parameters import *

# mf
from visualisations.mean_field_banana import visualise as visualise_mf_banana
from visualisations.mean_field_circle import visualise as visualise_mf_circle
from visualisations.mean_field_eight_schools import visualise as visualise_mf_eight_schools
from visualisations.mean_field_figure_eight import visualise as visualise_mf_figure_eight
from visualisations.mean_field_two_hills import visualise as visualise_mf_two_hills
# pf
from visualisations.planar_flows_banana import visualise as visualise_pf_banana
from visualisations.planar_flows_circle import visualise as visualise_pf_circle
from visualisations.planar_flows_eight_schools import visualise as visualise_pf_eight_schools
from visualisations.planar_flows_figure_eight import visualise as visualise_pf_figure_eight
from visualisations.planar_flows_two_hills import visualise as visualise_pf_two_hills


# Disable CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def visualise(q, model_choice, training_parameters):
    if model_choice == MEAN_FIELD:
        if training_parameters['name'] == 'two_hills':
            visualise_mf_two_hills(q, training_parameters['shape'])
        elif training_parameters['name'] == 'banana':
            visualise_mf_banana(q, training_parameters['shape'])
        elif training_parameters['name'] == 'circle':
            visualise_mf_circle(q, training_parameters['shape'])
        elif training_parameters['name'] == 'figure_eight':
            visualise_mf_figure_eight(q, training_parameters['shape'])
        elif training_parameters['name'] == 'eight_schools':
            visualise_mf_eight_schools(q, training_parameters['shape'])
    elif model_choice == PLANAR_FLOWS:
        if training_parameters['name'] == 'two_hills':
            visualise_pf_two_hills(q, training_parameters['shape'])
        elif training_parameters['name'] == 'banana':
            visualise_pf_banana(q, training_parameters['shape'])
        elif training_parameters['name'] == 'circle':
            visualise_pf_circle(q, training_parameters['shape'])
        elif training_parameters['name'] == 'figure_eight':
            visualise_pf_figure_eight(q, training_parameters['shape'])
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
    target = 'circle'

    """ Model for Inference | Possible Choices:
    mean_field
    planar_flows
    """
    model_choice = 'planar_flows'

    training_parameters = get_training_parameters(target)

    q = load_model(model_choice, training_parameters)

    visualise(q, model_choice, training_parameters)
