import numpy as np
import tensorflow as tf

from external.psis import psislw
from khat import q_posterior, log_joint_pdf
from models.model import get_model
from parameters import get_training_parameters, target_distributions
from train import train

target = 'eight_schools'
model_choice = 'planar_flows'

n_flows = [1, 2, 4, 8, 12, 16, 20]
n_flows = [1, 2]
K = 3

with open('khat_evolution_eight_schools.npy', 'rb') as f:
    khats = list(np.load(f))

print(f'Current Khats: {khats}')

n_flow = 8

print(f"- N FLOW: {n_flow}")
# Change shape in params
target_distributions[-1]['n_flows'] = n_flow
print(f"    New params: {target_distributions[-1]}")

training_parameters = get_training_parameters(target)

average_khat = 0

q = get_model(model_choice, training_parameters)

q = train(q, training_parameters)

for k in range(K):

    z, log_q = q_posterior(q, model_choice, training_parameters)

    log_p = log_joint_pdf(z, training_parameters)

    log_r = log_p - log_q

    _, khat = psislw(log_r)

    print(f"    - New khat: {khat}")

    average_khat += khat

khats.append(average_khat / K)

khats = np.array(khats)

with open('khat_evolution_eight_schools.npy', 'wb') as f:
    np.save(f, khats)





for n flow 
    
    train
    tf.Graph()

    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    for k in K

        khat 

        avg khat 

        append(avg_khat)