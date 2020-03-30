import os
import django

os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.orgel"
# os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.toy"


django.setup()

# Shell Plus Model Imports
from features.models import AtomDescriptor, BondDescriptor, ConnectivityMatrix, DistanceMatrix, Fingerprint, ProximityMatrix, SpeciesDescriptor, TrainingSet, Transformation
from guardian.models import GroupObjectPermission, UserObjectPermission
from django.contrib.contenttypes.models import ContentType
from neuralnet.models import ActiveLearningLoop, NetArchitecture, NetCommunity, NetFamily, NeuralNetwork, NnPotential, NnPotentialStats
from jobs.models import Job, JobConfig, WorkBatch
from django.contrib.admin.models import LogEntry
from django.contrib.auth.models import Group, Permission, User
from django.contrib.sessions.models import Session
from pgmols.models import (AtomBasis, BasisSet, Batch, Calc, Cluster,
                           Geom, Hessian, Jacobian, MDFrame, Mechanism, Method, Mol, MolGroupObjectPermission,
                           MolSet, MolUserObjectPermission, PathImage, ProductLink, ReactantLink, Reaction,
                           ReactionPath, ReactionType, SinglePoint, Species, Stoichiometry, Trajectory)
# Shell Plus Django Imports
from django.core.cache import cache
from django.db import transaction
from django.utils import timezone
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.conf import settings
from django.db.models import Avg, Case, Count, F, Max, Min, Prefetch, Q, Sum, When, Exists, OuterRef, Subquery

import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
import json


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from neuralnet.utils.gpu import gpu_wrapper

from sigopt import Connection
from sigopt.examples import franke_function
import pdb
import random
import copy


from nff.train.transfer import freeze_parameters, unfreeze_readout
from nff.train import loss
import GPUtil
from datetime import date






def make_path():
    today = date.today()
    variable = str(today.day) + str(today.month) + str(today.year)
    path = './sigopt/{}/'.format(variable)

    if os.path.isdir(path) == False:
        os.mkdir(path)
    else:
        print('Path already exists')
    return path

def get_main_datasets(first_id=953, last_id=984):
    
    last_dataset = Dataset.from_file('/home/saxelrod/engaging/models/dataset_{}.pth.tar'.format(last_id))
    first_dataset = Dataset.from_file('/home/saxelrod/engaging/models/dataset_{}.pth.tar'.format(first_id))
    
    return first_dataset, last_dataset

def make_large_small(first_dataset, last_dataset, max_num=500, en_cutoff=0.8):


    large_gap_data = last_dataset.copy()
    small_gap_data = last_dataset.copy()



    small_gap_props = {key: [] for key in last_dataset.props.keys()}
    i = 0
    for data in last_dataset:
        e1 = data['energy_1']
        e0 = data['energy_0']
        gap = (e1-e0).item() * 27.2 / 627.5


        if gap <= en_cutoff:
            for key, val in data.items():
                small_gap_props[key].append(val)
            i += 1
            
        if i == max_num:
            break

    small_gap_data.props = small_gap_props


    large_gap_props = {key: [] for key in last_dataset.props.keys()}
    i = 0
    for data in first_dataset:
        e1 = data['energy_1']
        e0 = data['energy_0']
        gap = (e1-e0).item() * 27.2 / 627.5
        if gap > en_cutoff:
            for key, val in data.items():
                large_gap_props[key].append(val) 
            i += 1
        if i == max_num:
            break

    large_gap_data.props = large_gap_props
    
    return small_gap_data, large_gap_data
    
def add_p(targets, results, g_0=0.2, p_cutoff=20):

    targets['gap'] = [targets['energy_1'][i] - targets['energy_0'][i]
                                for i in range(len(targets['energy_0']))]

    results['gap'] = [results['energy_1'][i] - results['energy_0'][i]
                                for i in range(len(results['energy_0']))]

    
    targ_gap =  torch.cat(targets['gap']) 
    res_gap =  torch.cat(results['gap']) 
    
    g_0_kcal = g_0 * 627.5 / 27.2
    
    targets['p'] = [100 * torch.exp(-targ_gap**2 / (2* g_0_kcal **2 ))] 
    results['p']= [100 * torch.exp(-res_gap**2 / (2* g_0_kcal **2 ))]
    
#     pdb.set_trace()
    valid_idx = [i for i, p in enumerate(torch.cat(targets['p'])) if p >= p_cutoff]
    targets['p'] = [item for i, item in enumerate(torch.cat(targets['p'])) if i in valid_idx]
    results['p'] = [item for i, item in enumerate(torch.cat(results['p'])) if i in valid_idx]


    
    
def make_loss(asgn):
    
    multi_loss_dict = {'mse': [{'coef': asgn['energy_0_loss'], 'params': {'key': 'energy_0'}},
           {'coef': asgn['energy_0_grad_loss'], 'params': {'key': 'energy_0_grad'}},
           {'coef': asgn['energy_1_loss'], 'params': {'key': 'energy_1'}},
           {'coef': asgn['energy_1_grad_loss'], 'params': {'key': 'energy_1_grad'}}],
                
          'zhu': [{'coef': 1,
            'params': {'loss_type': 'mse',
             'expected_gap': 0.00735,
             'lower_energy': 'energy_0',
             'upper_energy': 'energy_1'}}],
                       
          'zhu_grad': [{'coef': asgn["zhu_grad_loss"],
            'params': {'loss_type': 'mse',
             'expected_gap': 0.00735,
             'lower_energy': 'energy_0',
             'upper_energy': 'energy_1'}}]}
    
    loss_fn = loss.build_multi_loss(multi_loss_dict=multi_loss_dict)
    
    return loss_fn
    
    
def make_transfer_model(net_id):

    net = NnPotential.objects.get(pk=net_id)
    
    transfer_pk = net.transferfrom.id
    weightpath = net.traininghyperparams['mounted_weightpath']
    transfer_path = os.path.join(weightpath, str(transfer_pk))
    model = load_model(transfer_path)
    model = freeze_parameters(model)
    unfreeze_readout(model)
    
    return model

def get_meta_data(path, suggestion_id, optimizer, max_epochs):
    train_metrics = [
        metrics.MeanAbsoluteError('energy_0'),
        metrics.MeanAbsoluteError('energy_0_grad'),
        metrics.MeanAbsoluteError('energy_1'),
        metrics.MeanAbsoluteError('energy_1_grad')
    ]
    
    outdir = path + str(suggestion_id)
    
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    
    train_hooks = [
        hooks.MaxEpochHook(max_epochs),
        hooks.CSVHook(
            outdir,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(
            outdir,
            metrics=train_metrics,
            separator = ' | ',
            time_strf='%M:%S'
        ),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=30,
            factor=0.5,
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True
        )
    ]
    
    dic = {"train_hooks": train_hooks, "train_metrics": train_metrics, "outdir": outdir}
    return dic

def make_loaders(assignments, small_gap_data, large_gap_data, max_num=500, en_cutoff=0.8,
                 batch_size=50):
    
    random_data_pct = assignments['random_data_pct']
    random_num = int(random_data_pct / 100 * max_num)
    small_gap_num = max_num - random_num
    
    combined_props = {key: val[:random_num] for key, val in large_gap_data.props.items()}
    for key in combined_props.keys():
        combined_props[key] += small_gap_data.props[key][:small_gap_num]

            
    new_dataset = small_gap_data.copy()
    new_dataset.props = combined_props
    
    
    train, val, test = split_train_validation_test(new_dataset, val_size=0.2, test_size=0.2)
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_dicts)
    
    return train_loader, val_loader, test_loader

    
def evaluate_model(assignments,
                   net_id,
                   suggestion_id, 
                   batch_size, 
                   small_gap_data,
                   large_gap_data,
                   path,
                   max_num,
                   en_cutoff,
                   max_epochs,
                   g_0,
                   p_cutoff):
    
    train_loader, val_loader, test_loader = make_loaders(assignments=assignments,
                           small_gap_data=small_gap_data,
                           large_gap_data=large_gap_data,
                           max_num=max_num,
                           en_cutoff=en_cutoff)
    
    model = make_transfer_model(net_id)
    loss_fn = make_loss(assignments)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=10**(assignments['log_lr']))

    metadata = get_meta_data(path=path, suggestion_id=suggestion_id, optimizer=optimizer,
                            max_epochs=max_epochs)
    device = GPUtil.getAvailable(order='memory', limit=4, maxLoad=1.0,
                                   maxMemory=1.0)[0]
    T = Trainer(
        model_path=metadata["outdir"],
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=metadata["train_hooks"],
        mini_batches=1
    )

    T.train(device=device, n_epochs=max_epochs)

    
    results, targets, val_loss = evaluate(T.get_best_model(), test_loader, loss_fn, device=device,
                             offset=False, adj_type=None)
    
    
    add_p(targets=targets, results=results, g_0=g_0, p_cutoff=p_cutoff)
    
    # need to change this 
    
    val_loss = torch.mean(torch.abs(torch.stack(targets['p']) - torch.stack(results['p']))).item()
    
    return val_loss



def create_expt():
    conn = Connection(client_token="JQJLZYNHOWKBUXWMYBZFKRKHURZAZRIQWERJSBKWZUBODXEQ")

    experiment = conn.experiments().create(
        name='ci_opt',
        metrics=[dict(name='averaged_loss', objective='minimize')],
        parameters=[

            {"name": "zhu_grad_loss", "type": 'double', "bounds": {"min": 0, "max": 1}},
            {"name": "energy_0_grad_loss", "type": 'double', "bounds": {"min": 0, "max": 1}},
            {"name": "energy_1_grad_loss", "type": 'double', "bounds": {"min": 0, "max": 1}},
            {"name": "energy_0_loss", "type": 'double', "bounds": {"min": 0, "max": 1}},
            {"name": "energy_1_loss", "type": 'double', "bounds": {"min": 0, "max": 1}},
            {"name": "random_data_pct", "type": 'double', "bounds": {"min": 0, "max": 50}},
            {"name": "log_lr", "type": 'double', "bounds": {"min": -5, "max": -3}}


        ],

        # usually 10-20 x number of parameters
        observation_budget = 140, 
    )
    
    return conn, experiment

def run(first_id=953, last_id=984, max_num=1000, en_cutoff=0.8, batch_size=50,
       max_epochs=100, g_0=0.2, p_cutoff=20):
    
    conn, experiment = create_expt()
    first_dataset, last_dataset = get_main_datasets(first_id=first_id, last_id=last_id)
    small_gap_data, large_gap_data = make_large_small(first_dataset, last_dataset,
                                                     max_num=max_num,
                                                     en_cutoff=en_cutoff)
    path = make_path()
    min_value = float('inf')

    
    while experiment.progress.observation_count < experiment.observation_budget:
        
        suggestion = conn.experiments(experiment.id).suggestions().create()
        print(suggestion.assignments)
        
        try:     
            
            value = evaluate_model(assignments=suggestion.assignments,
                               net_id=last_id,
                               suggestion_id=suggestion.id, 
                               batch_size=batch_size, 
                               small_gap_data=small_gap_data,
                               large_gap_data=large_gap_data,
                               path=path,
                               max_num=max_num,
                               en_cutoff=en_cutoff,
                               max_epochs=max_epochs,
                               g_0=g_0,
                               p_cutoff=p_cutoff)


            print("\n")
            print (value)
            print("\n")
            if value < min_value:
                min_value = value
                best_params = dict(suggestion.assignments)
                
                with open('best_params.json', 'w') as f:
                    json.dump(best_params, f)

            conn.experiments(experiment.id).observations().create(
              suggestion=suggestion.id,
              value=value,
            )

            experiment = conn.experiments(experiment.id).fetch()
            
        except Exception as e:
            # raise e
            print(e)
            # pdb.post_mortem()
            continue
        except SystemExit as e:
            print(e)
            continue


if __name__ == "__main__":
    run()


            