"""Tools to build layers"""
import collections
from argparse import Namespace

import numpy as np
import torch

from torch.nn import ModuleDict, Sequential
from nff.nn.activations import shifted_softplus
from nff.nn.layers import Dense


layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "Dense": Dense,
    "shifted_softplus": shifted_softplus
}


def construct_sequential(layers):
    """Construct a sequential model from list of params 
    
    Args:
        layers (list): list to describe the stacked layer params. Example:
            layers = [
                {'name': 'linear', 'param' : {'in_features': 10, 'out_features': 20}},
                {'name': 'linear', 'param' : {'in_features': 10, 'out_features': 1}}
            ]
    
    Returns:
        Sequential: Stacked Sequential Model 
    """
    return Sequential(collections.OrderedDict(
            [layer['name'] + str(i), layer_types[layer['name']](**layer['param'])] 
            for i, layer in enumerate(layers)
    ))


def construct_module_dict(moduledict):
    """construct moduledict from a dictionary of layers
    
    Args:
        moduledict (dict): Description
    
    Returns:
        ModuleDict: Description
    """
    models = ModuleDict()
    for key in moduledict:
        models[key] = construct_sequential(moduledict[key])
    return models


def get_default_readout(n_atom_basis):
    """Default setting for readout layers. Predicts only the energy of the system.

    Args:
        n_atom_basis (int): number of atomic basis. Necessary to match the dimensions of
            the linear layer.

    Returns:
        DEFAULT_READOUT (dict)
    """

    DEFAULT_READOUT = {
        'energy': [
            {'name': 'linear', 'param' : { 'in_features': n_atom_basis, 'out_features': int(n_atom_basis / 2)}},
            {'name': 'shifted_softplus', 'param': {}},
            {'name': 'linear', 'param' : { 'in_features': int(n_atom_basis / 2), 'out_features': 1}}
        ]
    }

    return DEFAULT_READOUT

def torch_nbr_list(atomsobject, cutoff, device='cuda:0', directed=True):
    """Pytorch implementations of nbr_list for minimum image convention, the offsets are only limited to 0, 1, -1:
    it means that no pair interactions is allowed for more than 1 periodic box length. It is so much faster than 
    neighbor_list algorithm in ase.

    It is similar to the output of neighbor_list("ijS", atomsobject, cutoff) but a lot faster
    
    Args:
        atomsobject (TYPE): Description
        cutoff (float): cutoff for 
        device (str, optional): Description
    
    Returns:
        i, j, cutoff: just like ase.neighborlist.neighbor_list
    
    """
    xyz = torch.Tensor(atomsobject.get_positions() ).to(device)
    dis_mat = xyz[None, :, :] - xyz[:, None, :]
    cell_dim = torch.Tensor(atomsobject.get_cell()).diag().to(device)

    offsets = -dis_mat.ge(0.5 * cell_dim).to(torch.float) + dis_mat.lt(-0.5 * cell_dim).to(torch.float)
    dis_mat = dis_mat + offsets * cell_dim

    dis_sq = dis_mat.pow(2).sum(-1)
    mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)

    nbr_list = mask.nonzero()
    if directed:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    i, j  = nbr_list[:, 0].detach().to("cpu").numpy(), nbr_list[:, 1].detach().to("cpu").numpy()

    offsets = offsets[nbr_list[:, 0], nbr_list[:, 1], :].detach().to("cpu").numpy()

    return i, j, offsets 
