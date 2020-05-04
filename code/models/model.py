import sys
import tensorflow as tf

sys.path.append("..")

# Imports are from the code/
from models.model_mean_field import MeanField
from models.model_planar_flows import PlanarFlows
from models.model_radial_flows import RadialFlows
from parameters import existing_distributions, existing_models
from parameters import MEAN_FIELD, PLANAR_FLOWS, RADIAL_FLOWS, MODEL_SAVES_FOLDER, MODEL_SAVE_EXTENSION

def get_model(model_choice, training_parameters):
    assert (model_choice) in existing_models, f"Model chosen ({model_choice}) does not exist."

    if model_choice == MEAN_FIELD:
        model = MeanField(
            d=training_parameters['d'],
            shape=training_parameters['shape'],
        )
        model(tf.zeros(training_parameters['shape'])) # build model
    elif model_choice == PLANAR_FLOWS:
        model = PlanarFlows(
            d=training_parameters['d'],
            n_flows=training_parameters['n_flows'],
            shape=training_parameters['shape']
        )
        model(tf.zeros(training_parameters['shape'])) # build model
    elif model_choice == RADIAL_FLOWS:
        model = RadialFlows(
            d=training_parameters['d'],
            n_flows=training_parameters['n_flows'],
            shape=training_parameters['shape']
        )
        model(tf.zeros(training_parameters['shape'])) # build model

    return model

def save_model(q, model_choice, target, model_name_suffix=''):
    """
    Constitutes valid save filename and saves weights of the model under it.
    It is assumed that the model choice and target parameters have already been verified.
    """
    if model_name_suffix:
        model_save_filename = f"{MODEL_SAVES_FOLDER}/{model_choice}_{target}_{model_name_suffix}.{MODEL_SAVE_EXTENSION}"
    else:
        model_save_filename = f"{MODEL_SAVES_FOLDER}/{model_choice}_{target}.{MODEL_SAVE_EXTENSION}"
    q.save_weights(model_save_filename)
    print('-- Model Saved')

def load_model(model_choice, training_parameters, model_name_suffix=''):
    q = get_model(model_choice, training_parameters)

    if model_name_suffix:
        model_save_filename =  f"{MODEL_SAVES_FOLDER}/{model_choice}_{training_parameters['name']}_{model_name_suffix}.{MODEL_SAVE_EXTENSION}"
    else:
        model_save_filename =  f"{MODEL_SAVES_FOLDER}/{model_choice}_{training_parameters['name']}.{MODEL_SAVE_EXTENSION}"

    q.load_weights(model_save_filename)

    return q