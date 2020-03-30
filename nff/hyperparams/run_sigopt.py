import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import shutil
import sys
import json
from math import ceil


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from neuralnet.utils.gpu import gpu_wrapper

from sigopt import Connection
from sigopt.examples import franke_function
import pdb
import copy
from datetime import date


DIABATIC_KEYS = ['d0', 'd1', 'lambda']
ADIABATIC_KEYS = ['energy_0', 'energy_1']
RESULTS_FILE = 'results.json'
BEST_RESULTS_FILE = 'best_params.json'
BOOL_CATEGORIES = ['add_exc_energy', 'lower_e_mol_property', 'upper_e_mol_property', 'lambda_mol_property',
                   'schedule_loss', 'adiabatic']
# BOOL_CATEGORIES = ['add_exc_energy', 'lower_e_mol_property', 'upper_e_mol_property', 'lambda_mol_property']
# SET_ASSIGNMENTS = {'schedule_loss': 'False', 'adiabatic': 'False'}

SET_ASSIGNMENTS = {}
CORRESPONDENCE_KEYS = {"energy_0_ad": "energy_0",
                      "energy_1_ad": "energy_1",
                      "energy_0_ad_grad": "energy_0_grad",
                      "energy_1_ad_grad": "energy_1_grad"}

N_EPOCHS = 15

LR = 3e-4
# will this be a problem?
TRANSITION_TIME = 7


def make_time_functions(transition_time, assignments):
    tau = transition_time / 3
    out = {"energy_0_ad": lambda t: 0.1 * np.exp(-t/tau),
                  "energy_1_ad": lambda t: 0.1 * np.exp(-t/tau),
                  "energy_0_ad_grad": lambda t: np.exp(-t/tau),
                  "energy_1_ad_grad": lambda t: np.exp(-t/tau),
                  "energy_0": lambda t: 0.1 * (1 - np.exp(-t/tau)),
                  "energy_1": lambda t: 0.1 * (1 - np.exp(-t/tau)),
                  "energy_0_grad": lambda t: (1 - np.exp(-t/tau)),
                  "energy_1_grad": lambda t: (1 - np.exp(-t/tau)),
                  "d0_grad": lambda t: (1 - np.exp(-t/tau)) * assignments['d0_loss'],
                  "d1_grad": lambda t: (1 - np.exp(-t/tau)) * assignments['d1_loss'],
                  "lambda_grad": lambda t: (1 - np.exp(-t/tau)) * assignments['lambda_loss'],
               
                 }
    return out 

def update_dataset(dataset, add_adiabatic=False, diabatic_keys=True):
    init = [torch.zeros_like(item) for item in dataset.props['energy_0']]
    
    if diabatic_keys:
        dataset.props.update({'d0_grad': init})
        dataset.props.update({'d1_grad': init})
        dataset.props.update({'lambda_grad': init})
    
    if add_adiabatic:
        dataset.props.update({'energy_0_ad_grad': init})
        dataset.props.update({'energy_1_ad_grad': init})
        dataset.props.update({'energy_0_ad': init})
        dataset.props.update({'energy_1_ad': init})
    
def make_bool(names):
    params = []
    for name in names:
        dic = {
                  "name": name,
                  "type": "categorical",
                  "categorical_values": ['True', 'False']
                }

        params.append(dic)
       

    return params

def make_atomic_readout(assignments):
    
    n_atom_basis = assignments['n_atom_basis']
    atom_dropout = assignments['atom_dropout']
    layers = [{'name': 'Dropout', 'param': {'p': atom_dropout}},
              {'name': 'linear', 'param' : { 'in_features': n_atom_basis, 
                                            'out_features': ceil(n_atom_basis/2)}},
                {'name': 'shifted_softplus', 'param': {}},
                {'name': 'Dropout', 'param': {'p': atom_dropout}},
                {'name': 'linear', 'param' : { 'in_features': ceil(n_atom_basis/2), 
                                                'out_features': 1}}]

        
    if eval(assignments['adiabatic']):
        atomic_keys = ADIABATIC_KEYS
    else:
        atomic_keys = DIABATIC_KEYS
        
    if eval(assignments['schedule_loss']) and not eval(assignments['adiabatic']):
        atomic_keys = [*atomic_keys, *['energy_0_ad', 'energy_1_ad']]
        
        
    new_atomic_keys = copy.deepcopy(atomic_keys)
    for assignment_key in assignments.keys():
        for atomic_key in atomic_keys:
            base_cond =  assignment_key.endswith('mol_property')
            cond_2 = assignment_key.startswith(atomic_key)
            cond_3 = atomic_key == 'energy_0' and assignment_key.startswith('lower_e')
            cond_4 = atomic_key == 'energy_1' and assignment_key.startswith('upper_e')
            cond_5 = atomic_key == 'd0' and assignment_key.startswith('lower_e')
            cond_6 = atomic_key == 'd1' and assignment_key.startswith('upper_e')
            
            if base_cond and any((cond_2, cond_3, cond_4, cond_5, cond_6)):
                logic = eval(assignments[assignment_key])
                if logic:
                    new_atomic_keys.remove(atomic_key)
                
    atomic_readout = {key: layers for key in new_atomic_keys}
    return atomic_readout

    
def make_mol_features(assignments):
    
    n_atom_basis = assignments['n_atom_basis']
    n_mol_basis = assignments['n_mol_basis']
    num_layers = assignments['mol_feature_layers']
    
    fracs = np.linspace(0, 1, num_layers + 1)
    layers = []
    for layer in range(num_layers):
        if layer != 0:
            layers.append({'name': 'shifted_softplus', 'param': {}})
        
        num_in = ceil((1-fracs[layer])*n_atom_basis + fracs[layer]*n_mol_basis)
        num_out = ceil((1-fracs[layer+1])*n_atom_basis + fracs[layer+1]*n_mol_basis)
        layers.append({'name': 'linear', 'param': {'in_features': num_in,
                                                  'out_features': num_out}})
        
    
    return layers
    
def make_mol_readout(assignments):
    
    n_mol_basis = assignments['n_mol_basis']
    mol_dropout = assignments['mol_dropout']

    layers = [{'name': 'Dropout', 'param': {'p': mol_dropout}},
              {'name': 'linear', 'param': {'in_features': n_mol_basis,
                                            'out_features': ceil(n_mol_basis/2)}},
              {'name': 'shifted_softplus', 'param': {}},
              {'name': 'Dropout', 'param': {'p': mol_dropout}},
              {'name': 'linear', 'param': {'in_features': ceil(n_mol_basis/2),
                                        'out_features': 1}}
                ]
    
    if eval(assignments['adiabatic']):
        atomic_keys = ADIABATIC_KEYS
    else:
        atomic_keys = DIABATIC_KEYS
        
    mol_keys = []
    for assignment_key in assignments.keys():
        for atomic_key in atomic_keys:
            
            base_cond =  assignment_key.endswith('mol_property')
            cond_2 = assignment_key.startswith(atomic_key)
            cond_3 = atomic_key == 'energy_0' and assignment_key.startswith('lower_e')
            cond_4 = atomic_key == 'energy_1' and assignment_key.startswith('upper_e')
            cond_5 = atomic_key == 'd0' and assignment_key.startswith('lower_e')
            cond_6 = atomic_key == 'd1' and assignment_key.startswith('upper_e')
            
            if base_cond and any((cond_2, cond_3, cond_4, cond_5, cond_6)):
                logic = eval(assignments[assignment_key])
                if logic:
                    mol_keys.append(atomic_key)
    molecular_readout = {key: layers for key in mol_keys}
    return molecular_readout

def make_coupling(assignments):
    if not eval(assignments['adiabatic']):
        coupling = {
            'inputs': DIABATIC_KEYS,
            'layers': [{'name': 'Diagonalize', 'param': {}}],
            'outputs': ADIABATIC_KEYS
        }
    else:
        coupling = {}
    
    return coupling

def make_add_results(assignments):
    if eval(assignments['add_exc_energy']):
        if eval(assignments['adiabatic']):
            add_results = {'energy_1': ['energy_0']}
        else:
            add_results = {'d1': ['d0']}
    else:
        add_results = {}

    return add_results

def make_params(assignments, batch_size):
    
    """
    Example:
        assignments = {
                        "add_exc_energy": True,
                        "d0_moel_property": False,
                        "d1_mol_property": True,
                        "lambda_mol_property": False,
                        "mol_dropout": 0.5,
                        "atom_dropout": 0.2,


        }
    
    """

    atomic_readout = make_atomic_readout(assignments)
    molecular_features = make_mol_features(assignments)
    molecular_readout = make_mol_readout(assignments)
    coupling = make_coupling(assignments)
    add_results = make_add_results(assignments)
    
    
    readoutdict = {
        'mol_readout': molecular_readout,
        'atom_readout': atomic_readout,
        'mol_features': molecular_features,
        'coupling': coupling,
        'add_results': add_results,
    }

    params = {
        'n_atom_basis': assignments['n_atom_basis'],
        'n_filters': assignments['n_filters'],
        'n_gaussians': assignments['n_gaussians'],
        'n_convolutions': assignments['n_convolutions'] ,
        'cutoff': assignments['cutoff'],
        'trainable_gauss': True, 
        'readoutdict': readoutdict,
        'batch_size': batch_size,
    }
    
    return params

def make_train_loss_fn(assignments):
    loss_coef = {'energy_0': 0.1, 'energy_0_grad': 1,
                'energy_1': 0.1, 'energy_1_grad': 1}
    if not eval(assignments['adiabatic']):
        loss_coef.update({
                'd0_grad': assignments['d0_loss'],
                'd1_grad': assignments['d1_loss'],
                'lambda_grad': assignments['lambda_loss']})
    
    loss_fn = loss.build_mse_loss(loss_coef)
    return loss_fn
    
def make_eval_loss_fn():
    loss_coef = {'energy_0': 0.1, 'energy_0_grad': 1,
                'energy_1': 0.1, 'energy_1_grad': 1}
#     loss_coef = {'energy_0': 1, 'energy_1': 1}
    loss_fn = loss.build_mse_loss(loss_coef)
    return loss_fn



def create_model(assignments, suggestion_id, batch_size, device, 
                 train, val, test, path, i=0):
    
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=5, collate_fn=collate_dicts)

    params = make_params(assignments, batch_size)


    print("Assignments: {}".format(assignments))
    print("\n")
    print("Params: {}".format(params))
    print("\n")

#     mini_batches = ceil(100/batch_size)
    mini_batches = 1
    model = get_model(params)

    loss_fn = make_train_loss_fn(assignments)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=assignments['lr'])
#     pdb.set_trace()
#    optimizer = Adam(trainable_params, lr=LR)


    train_metrics = [
        metrics.MeanAbsoluteError('energy_0'),
        metrics.MeanAbsoluteError('energy_0_grad'),
        metrics.MeanAbsoluteError('energy_1'),
        metrics.MeanAbsoluteError('energy_1_grad')
    ]
    
    OUTDIR = path + str(suggestion_id)
    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)
    
    train_hooks = [
        hooks.MaxEpochHook(100),
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
            patience=30,
            factor=0.5,
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True
        )
    ]
  
    if eval(assignments['schedule_loss']) and not eval(assignments['adiabatic']):
        
        time_functions = make_time_functions(TRANSITION_TIME, assignments)
        
        train_hooks.append(hooks.LossSchedulingHook(
            time_functions=time_functions,
            per_epoch=True,
            correspondence_keys=CORRESPONDENCE_KEYS
            ))


    print(model)
    
    T = Trainer(
        model_path=OUTDIR,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks,
        mini_batches=mini_batches
    )

    T.train(device=device, n_epochs=N_EPOCHS)
    
    
    eval_loss_fn = make_eval_loss_fn()
    _, _, val_loss = evaluate(T.get_best_model(), test_loader, eval_loss_fn, device=device,
                             offset=True, adj_type='mean')
    
    params.update({key: val for key, val in assignments.items() if key not in params.keys()})
    
    return val_loss, params

@gpu_wrapper
def evaluate_model(assignments, i, suggestion_id, batch_size=None, device=None,
                  train=None, val=None, test=None, path=None):
    value, params = create_model(assignments=assignments, i=i, suggestion_id=suggestion_id, batch_size=batch_size,
                        device=device, train=train, val=val, test=test, path=path)
    return value, params


def save_params(value, params, suggestion):
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = []
    item = {"loss": value, "params": params, "assignments": dict(suggestion.assignments)}
    if item not in results:
        results.append(item)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


def main():
    base_dataset = Dataset.from_file('excited_sigopt_training.pth.tar')
#    test = Dataset.from_file('excited_sigopt_validation_no_outliers.pth.tar')


    today = date.today()
    variable = str(today.day) + str(today.month) + str(today.year)
    path = './sigopt/{}/'.format(variable)

    if os.path.isdir(path) == False:
        os.mkdir(path)
    else:
        print('Path already exists')



    conn = Connection(client_token="KTNMWLZQYQSNCHVHPGIWSAVXEWLEWABZAHIJOLXKWAHQDRQE")


    experiment = conn.experiments().create(
        name='tutorial ',
        metrics=[dict(name='energy', objective='minimize')],
        parameters=[
            dict(name='n_atom_basis', type='int', bounds=dict(min=100, max=300)),
            dict(name='n_mol_basis', type='int', bounds=dict(min=100, max=300)),
            dict(name='mol_feature_layers', type='int', bounds=dict(min=2, max=4)),
            dict(name='n_filters', type='int', bounds=dict(min=100, max=300)),
            dict(name='n_gaussians', type='int', bounds=dict(min=30, max=70)),
            dict(name='n_convolutions', type='int', bounds=dict(min=4, max=7)),
            dict(name='cutoff', type='double', bounds=dict(min=4.0, max=7.0)),
            dict(name='mol_dropout', type='double', bounds=dict(min=0.0, max=0.5)),
            dict(name='atom_dropout', type='double', bounds=dict(min=0.0, max=0.5)),
            dict(name='d0_loss', type='double', bounds=dict(min=0.0, max=0.05)),
            dict(name='d1_loss', type='double', bounds=dict(min=0.0, max=0.05)),
            dict(name='lambda_loss', type='double', bounds=dict(min=0.0, max=0.05)),
            dict(name='lr', type='double', bounds=dict(min=2.5e-5, max=3e-4)),
            *make_bool(BOOL_CATEGORIES)

        ],


        observation_budget = 360,


    )


    i = 0


    min_value = float('inf')
    best_params = {}
    conn.experiments(experiment.id).suggestions().delete()

    while experiment.progress.observation_count < experiment.observation_budget:

        suggestion = conn.experiments(experiment.id).suggestions().create()
        assignments = suggestion.assignments
        assignments.update(SET_ASSIGNMENTS)

        dataset = base_dataset.copy()
        if eval(assignments['schedule_loss']) and not eval(assignments['adiabatic']):
            add_adiabatic = True
        else:
            add_adiabatic = False
        if eval(assignments['adiabatic']):
            add_diabatic_keys = False
        else:
            add_diabatic_keys = True

        update_dataset(dataset, add_adiabatic, add_diabatic_keys)
        train, val, test = split_train_validation_test(dataset, val_size=0.1, test_size=0.1)



        try:
            value, params  = evaluate_model(assignments=assignments, i=i, suggestion_id=suggestion.id,
                                           train=train, test=test, val=val, path=path)

            print("\n")
            print (value)
            print("\n")
            save_params(value.item(), params, suggestion)

            if value < min_value:
                min_value = value
                best_params = params
                with open(BEST_RESULTS_FILE, 'w') as f:
                    json.dump(best_params, f, indent=4, sort_keys=True)

            conn.experiments(experiment.id).observations().create(
              suggestion=suggestion.id,
              value=value,
            )


            experiment = conn.experiments(experiment.id).fetch()

        except Exception as e:
            print(e)
#             continue
            pdb.post_mortem()
        except SystemExit as e:
            print(e)
            conn.experiments(experiment.id).observations().create(
                  failed=True,
                  suggestion=suggestion.id
                  )

            continue


if __name__ == "__main__":
    main()


