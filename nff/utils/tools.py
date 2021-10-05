"""Assorted tools in the package.
Adapted from https://github.com/atomistic-machine-learning/schnetpack/blob/dev/src/schnetpack/utils/spk_utils.py
"""
import json
import logging
import collections
from argparse import Namespace

import numpy as np
import torch

from torch.nn import ModuleDict, Sequential
from nff.nn.activations import (shifted_softplus, Swish,
                                LearnableSwish)
from nff.nn.layers import Dense


__all__ = [
    "set_random_seed",
    "compute_params",
    "to_json",
    "read_from_json",
]

layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "Dense": Dense,
    "shifted_softplus": shifted_softplus,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU":  torch.nn.ELU,
    "swish": Swish,
    "learnable_swish": LearnableSwish,
    "softplus": torch.nn.Softplus
}


def construct_Sequential(layers):
    """Construct a sequential model from list of params 

    Args:
        layers (list): list to describe the stacked layer params 
                        example:    [
                                        {'name': 'linear', 'param' : {'in_features': 10, 'out_features': 20}},
                                        {'name': 'linear', 'param' : {'in_features': 10, 'out_features': 1}}
                                    ]

    Returns:
        Sequential: Stacked Sequential Model 
    """

    return Sequential(collections.OrderedDict([layer['name']+str(i),
                                               layer_types[layer['name']](
                                                   **layer['param'])
                                               ] for i, layer in enumerate(layers)))


def construct_ModuleDict(moduledict):
    """construct moduledict from a dictionary of layers

    Args:
        moduledict (dict): Description

    Returns:
        ModuleDict: Description
    """
    models = ModuleDict()
    for key in moduledict:
        models[key] = construct_Sequential(moduledict[key])
    return models


def set_random_seed(seed):
    """
    This function sets the random seed (if given) or creates one for torch and numpy random state initialization

    Args:
        seed (int, optional): if seed not present, it is generated based on time
    """
    import time
    import numpy as np

    # 1) if seed not present, generate based on time
    if seed is None:
        seed = int(time.time() * 1000.0)
        # Reshuffle current time to get more different seeds within shorter time intervals
        # Taken from https://stackoverflow.com/questions/27276135/python-random-system-time-seed
        # & Gets overlapping bits, << and >> are binary right and left shifts
        seed = (
            ((seed & 0xFF000000) >> 24)
            + ((seed & 0x00FF0000) >> 8)
            + ((seed & 0x0000FF00) << 8)
            + ((seed & 0x000000FF) << 24)
        )
    # 2) Set seed for numpy (e.g. splitting)
    np.random.seed(seed)
    # 3) Set seed for torch (manual_seed now seeds all CUDA devices automatically)
    torch.manual_seed(seed)
    logging.info("Random state initialized with seed {:<10d}".format(seed))


def compute_params(model):
    """
    This function gets a model as an input and computes its trainable parameters

    Args:
        model (AtomisticModel): model for which you want to compute the trainable parameters

    Returns:
        params (int): number of trainable parameters for the model
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def to_json(jsonpath, argparse_dict):
    """
    This function creates a .json file as a copy of argparse_dict

    Args:
        jsonpath (str): path to the .json file
        argparse_dict (dict): dictionary containing arguments from argument parser
    """
    with open(jsonpath, "w") as fp:
        json.dump(argparse_dict, fp, sort_keys=True, indent=4)


def read_from_json(jsonpath):
    """
    This function reads args from a .json file and returns the content as a namespace dict

    Args:
        jsonpath (str): path to the .json file

    Returns:
        namespace_dict (Namespace): namespace object build from the dict stored into the given .json file.
    """
    with open(jsonpath) as handle:
        dict = json.loads(handle.read())
        namespace_dict = Namespace(**dict)
    return namespace_dict


def make_directed(nbr_list):

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed

def make_undirected(nbr_list):
    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if not directed:
        return nbr_list, directed
    nbrs = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]
    
    return nbrs, directed
