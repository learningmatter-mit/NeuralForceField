
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
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-data", type=str, default='ethanol_ccsd')
parser.add_argument("-id", type=int, default=None)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    n_epochs = 2
    n_obs = 2
else:
    token = 'RXGPHWIUAMLHCDJCDBXEWRAUGGNEFECMOFITCRHCEOBRMGJU'
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
            dict(name='n_rbf', type='int', bounds=dict(min=2, max=6)),  # 4->64
            dict(name='envelope_p', type='int', bounds=dict(min=2, max=3)),  # 4->8
            dict(name='n_spher', type='int', bounds=dict(min=2, max=3)),  # 4->8
            dict(name='l_spher', type='int', bounds=dict(min=2, max=3)),  # 4->8
            dict(name='embed_dim', type='int', bounds=dict(min=6, max=9)),  # 64->512
            dict(name='batch_size', type='int', bounds=dict(min=2, max=6)),  # 4->64
            dict(name='n_bilinear', type='int', bounds=dict(min=2, max=4)),  # 4->16
            dict(name='cutoff', type='double', bounds=dict(min=3.5, max=6.0), precision=1),
            dict(name='activation', type='categorical', categorical_values= ["Tanh" ,"ReLU" ,"shifted_softplus" ,
                                                                            "sigmoid"  ,"Dropout"  ,"LeakyReLU",
                                                                            "ELU" ,"swish"]),
            dict(name='n_convolutions', type='int', bounds=dict(min=2, max=7)),
        ],
        observation_budget=n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()


while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    trainparam = suggestion.assignments
    trainparam['n_rbf'] = 2**trainparam['n_rbf']
    trainparam['envelope_p'] = 2**trainparam['envelope_p']
    trainparam['n_spher'] = 2**trainparam['n_spher']
    trainparam['l_spher'] = 2**trainparam['l_spher']
    trainparam['embed_dim'] = 2**trainparam['embed_dim']
    trainparam['batch_size'] = 2**trainparam['batch_size']
    trainparam['n_bilinear'] = 2**trainparam['n_bilinear']
    print(trainparam)

    test_mae = train(params, trainparam, suggestion.id, ForceDime, n_epochs, angle=True)
    # updat result to server
    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=test_mae,
    )

    experiment = conn.experiments(experiment.id).fetch()
