
import argparse
from sigopt import Connection
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../")
import os
import shutil

from train import train
from forcedime import ForceDime

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str, default='mkxu/fdime_ethanol')
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-data", type=str, default='ethanol_dft')
parser.add_argument("-id", type=int, default=None)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_epochs = 2
    n_obs = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_epochs = 2000 
    n_obs = 1000


# Generate parameter range 
logdir = params['logdir']

#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=logdir,
        metrics=[dict(name='loss', objective='minimize')],
        parameters=[
            dict(name='lr', type='double', bounds=dict(min=1e-4, max=1e-3), transformation="log"),
            # 2*x
            dict(name='n_rbf', type='int', bounds=dict(min=3, max=8), default=3),  # 6->16
            dict(name='n_spher', type='int', bounds=dict(min=3, max=8), default=3),  # 6->16
            dict(name='l_spher', type='int', bounds=dict(min=3, max=8), default=3),  # 6->16
            dict(name='n_bilinear', type='int', bounds=dict(min=4, max=8), default=4),  # 6->16
            dict(name='envelope_p', type='int', bounds=dict(min=3, max=8), default=3),  # 6->16
            # 2**x
            dict(name='embed_dim', type='int', bounds=dict(min=7, max=8), default=7),  # 128->256
            dict(name='batch_size', type='int', bounds=dict(min=3, max=5), default=5),  # 8->32
            # non-processed
            dict(name='cutoff', type='double', bounds=dict(min=4.5, max=7.0), default=5.0, precision=1),
            dict(name='activation', type='categorical', categorical_values= ["shifted_softplus","swish"], default="swish"),
            # "Tanh", "ReLU", "shifted_softplus", "sigmoid", "Dropout", "LeakyReLU", "ELU", "swish"
            dict(name='n_convolutions', type='int', bounds=dict(min=3, max=7), default=6),
        ],
        observation_budget=n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()


while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    trainparam = suggestion.assignments
    # times
    trainparam['n_rbf'] = 2*trainparam['n_rbf']
    trainparam['envelope_p'] = 2*trainparam['envelope_p']
    trainparam['n_spher'] = 2*trainparam['n_spher']
    trainparam['l_spher'] = 2*trainparam['l_spher']
    trainparam['n_bilinear'] = 2*trainparam['n_bilinear']
    # exponential
    trainparam['embed_dim'] = 2**trainparam['embed_dim']
    trainparam['batch_size'] = 2**trainparam['batch_size']
    print(trainparam)

    test_mae = train(params, trainparam, suggestion.id, ForceDime, n_epochs, angle=True)
    # updat result to server
    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=test_mae,
    )

    experiment = conn.experiments(experiment.id).fetch()
