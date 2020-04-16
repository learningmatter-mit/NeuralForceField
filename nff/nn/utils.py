"""Tools to build layers"""
import collections
from argparse import Namespace

import numpy as np
import torch

from torch.nn import ModuleDict, Sequential
from nff.nn.activations import shifted_softplus
from nff.nn.layers import Dense
from nff.utils.scatter import scatter_add


layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "Dense": Dense,
    "shifted_softplus": shifted_softplus,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout
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
    xyz = torch.Tensor(atomsobject.get_positions(wrap=True) ).to(device)
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

def chemprop_msg_update(h, nbrs):

    # nbr_dim x nbr_dim matrix, e.g. for nbr_dim = 4, all_idx =
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    all_idx = torch.stack([torch.arange(0, len(nbrs))] * len(nbrs)).long()

    # The first argument gives nbr list indices for which the second
    # neighbor of the nbr element matches the second neighbor of this element.
    # The second argument makes you ignore nbr elements equal to this one.
    # Example:
    # nbrs = [[1,2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]].
    # message = [m_{12}, m_{21}, m_{23}, m_{32}, m_{24}, m_{42}]
    # Then m_{12} = h_{32} + h_{42} (and not + h_{12})

    mask = (nbrs[:, 1] == nbrs[:, 1, None]) * (nbrs[:, 0] != nbrs[:, 0, None])

    # select the values of all_idx that are allowed by `mask`
    good_idx = all_idx[mask]

    # get the h's of these indices
    h_to_add = h[good_idx]

    # number of nbr_list matches for each index of `message`.
    # E.g. for the first index, with m_{12}, we got two matches

    num_matches = mask.sum(1).tolist()
    # map from indices `h_to_add` to the indices of `message`
    match_idx = torch.cat([torch.LongTensor([index] * match)
                           for index, match in enumerate(num_matches)])
    match_idx = match_idx.to(h.device)

    graph_size = h.shape[0]

    message = scatter_add(src=h_to_add,
                          index=match_idx,
                          dim=0,
                          dim_size=graph_size)

    return message


def chemprop_msg_to_node(h, nbrs, num_nodes):

    node_idx = torch.arange(num_nodes).to(h.device)
    nbr_idx = torch.arange(len(nbrs)).to(h.device)
    node_nbr_idx = torch.stack([nbr_idx] * len(node_idx))

    mask = (nbrs[:, 0] == node_idx[:, None])
    num_matches = mask.sum(1).tolist()
    match_idx = torch.cat([torch.LongTensor([index] * match)
                           for index, match in enumerate(num_matches)])
    match_idx = match_idx.to(h.device)

    good_idx = node_nbr_idx[mask]
    h_to_add = h[good_idx]

    node_features = scatter_add(src=h_to_add,
                                index=match_idx,
                                dim=0,
                                dim_size=num_nodes)

    return node_features

