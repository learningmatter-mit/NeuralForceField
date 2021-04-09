
import argparse
from sigopt import Connection
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../")
import os
import shutil

from train import train
# from forceconv import *
from forcepai import ForcePai

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str, default='mkxu/fpai_ethanol')
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-data", type=str, default='ethanol_dft')
parser.add_argument("-epoch", type=int, default=4000)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    params['epoch'] = 2
    n_obs = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
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
            dict(name='feat_dim', type='int', bounds=dict(min=64, max=256)),
            dict(name='num_conv', type='int', bounds=dict(min=2, max=5)),
            dict(name='n_rbf', type='int', bounds=dict(min=8, max=64)),
            dict(name='batch_size', type='int', bounds=dict(min=8, max=64)),
            dict(name='cutoff', type='double', bounds=dict(min=4.0, max=8.0)),
            dict(name='lr', type='double', bounds=dict(min=1e-6, max=1e-2), transformation="log"),
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

    test_mae = train(params, suggestion, ForcePai, params['epoch'])
    # updat result to server
    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=test_mae,
    )

    experiment = conn.experiments(experiment.id).fetch()

