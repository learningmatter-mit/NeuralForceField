
import argparse
from sigopt import Connection
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../")
import os
import shutil

from train import train
from forceconv import *

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
            dict(name='n_atom_basis', type='int', bounds=dict(min=32, max=512)),
            dict(name='n_edge_basis', type='int', bounds=dict(min=32, max=512)),
            dict(name='n_filters', type='int', bounds=dict(min=32, max=256)),
            dict(name='n_convolutions', type='int', bounds=dict(min=3, max=6)),
            dict(name='n_gaussians', type='int', bounds=dict(min=8, max=512)),
            dict(name='batch_size', type='int', bounds=dict(min=16, max=64)),
            dict(name='cutoff', type='double', bounds=dict(min=2.0, max=6.0)),
            dict(name='lr', type='double', bounds=dict(min=1e-4, max=1e-3)),
            dict(name='edge_filter_depth', type='int', bounds=dict(min=0, max=1)),
            dict(name='atom_filter_depth', type='int', bounds=dict(min=0, max=1)),
            dict(name='edge_update_depth', type='int', bounds=dict(min=0, max=1)),
            dict(name='atom_update_depth', type='int', bounds=dict(min=0, max=1)),
        ],
        observation_budget=n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()


while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    trainparam = suggestion.assignments

    print(trainparam)

    test_mae = train(params, suggestion, ForceConvolve, n_epochs)
    # updat result to server
    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=test_mae,
    )

    experiment = conn.experiments(experiment.id).fetch()

