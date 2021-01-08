
import argparse
from sigopt import Connection
from forceconv import *
import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../")
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from MD17data import *

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
            dict(name='n_gaussains', type='int', bounds=dict(min=32, max=128)),
            dict(name='n_atom_basis', type='int', bounds=dict(min=32, max=128)),
            dict(name='n_edge_basis', type='int', bounds=dict(min=32, max=128)),
            dict(name='n_filters', type='int', bounds=dict(min=32, max=128)),
            dict(name='n_convolutions', type='int', bounds=dict(min=3, max=6)),
            dict(name='n_gaussians', type='int', bounds=dict(min=32, max=64)),
            dict(name='batch_size', type='int', bounds=dict(min=8, max=64)),
            dict(name='cutoff', type='double', bounds=dict(min=2.0, max=5.0)),
            dict(name='lr', type='double', bounds=dict(min=1e-4, max=1e-3)),
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

    # get data 
    data = get_MD17data('ethanol_ccsd')
    dataset = pack_MD17data(data, 2000)

    train, val, test = split_train_validation_test(dataset, val_size=0.25, test_size=0.25)
    train_loader = DataLoader(train, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)

    print("building models")
    model = ForceConvolve(n_convolutions=trainparam['n_convolutions'], 
                          n_edge_basis=trainparam['n_edge_basis'], 
                          n_atom_basis=trainparam['n_atom_basis'], 
                          n_filters=trainparam['n_filters'], 
                          n_gaussians=trainparam['n_gaussians'], 
                          cutoff=trainparam['cutoff'])

    loss_fn = loss.build_mse_loss(loss_coef={'energy_grad': 1})
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=trainparam['lr'])
    train_metrics = [
        metrics.MeanAbsoluteError('energy_grad')
    ]

    DEVICE = params['device']
    OUTDIR = './{}/{}/sandbox'.format(params['logdir'], suggestion.id)

    print(OUTDIR)

    if os.path.exists(OUTDIR):
        newpath = os.path.join(os.path.dirname(OUTDIR), 'backup')
        if os.path.exists(newpath):
            shutil.rmtree(newpath)
        shutil.move(OUTDIR, newpath)

    train_hooks = [
        hooks.MaxEpochHook(n_epochs),
        hooks.CSVHook(
            OUTDIR,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(
            OUTDIR,
            metrics=train_metrics,
            separator = ' | ',
            time_strf='%M:%S'
        ),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=20,
            factor=0.5,
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True
        )
    ]

    T = Trainer(
        model_path=OUTDIR,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks
    )


    print("Start training ")
    T.train(device=DEVICE, n_epochs=n_epochs)

    # evaluate model 
    results, targets, val_loss = evaluate(T.get_best_model(), test_loader, loss_fn, device=DEVICE)
    key = 'energy_grad'
    pred = torch.stack(results[key][:-1], dim=0).view(-1).detach().cpu().numpy()
    targ = torch.stack(targets[key][:-1], dim=0).view(-1).detach().cpu().numpy()
    mae = np.abs(pred - targ).mean()

    print("Test loss for {}: {} kcal/(mol A)".format(params['data'], mae))

    # updat result to server
    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=mae,
    )

    experiment = conn.experiments(experiment.id).fetch()

