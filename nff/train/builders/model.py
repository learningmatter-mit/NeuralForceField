"""Helper functions to create models, functions and other classes
    while checking for the validity of hyperparameters.
"""
import os
import numpy as np
import torch
from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.conformers import WeightedConformers
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.models.cp3d import ChemProp3D, OnlyBondUpdateCP3D
from nff.nn.models.dimenet import DimeNet, DimeNetDiabat, DimeNetDiabatDelta, DimeNetDelta

PARAMS_TYPE = {"SchNet":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'bond_par': float,
                   'trainable_gauss': bool,
                   'box_size': np.array,
                   'dropout_rate': float
               },
               "HybridGraphConv":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'mol_n_convolutions': int,
                   'mol_n_cutoff': float,
                   'sys_n_convolutions': int,
                   'sys_n_cutoff': float,
                   'V_ex_power': int,
                   'V_ex_sigma': float,
                   'trainable_gauss': bool
               },

               "WeightedConformers":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'trainable_gauss': bool,
                   'dropout_rate': float,
                   'readoutdict': dict,
                   'mol_fp_layers': list
               },

               "SchNetFeatures":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'bond_par': float,
                   'trainable_gauss': bool,
                   'box_size': np.array,
                   'dropout_rate': float,
                   'n_bond_hidden': int,
                   'n_bond_features': int,
                   'activation': str
               },

               "ChemProp3D":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'bond_par': float,
                   'trainable_gauss': bool,
                   'box_size': np.array,
                   'dropout_rate': float,
                   'cp_input_layers': list,
                   'schnet_input_layers': list,
                   'output_layers': list,
                   'n_bond_hidden': int,
                   'activation': str
               },

               "OnlyBondUpdateCP3D":

               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'bond_par': float,
                   'trainable_gauss': bool,
                   'box_size': np.array,
                   'schnet_dropout': float,
                   'cp_dropout': float,
                   'input_layers': list,
                   'output_layers': list,
                   'n_bond_hidden': int,
                   'activation': str
               },

               "DimeNet":
               {
                   "n_rbf": int,
                   "cutoff": float,
                   "envelope_p": int,
                   "n_spher": int,
                   "l_spher": int,
                   "atom_embed_dim": int,
                   "n_bilinear": int,
                   "activation": str,
                   "n_convolutions": int,
                   "output_keys": list,
                   "grad_keys": list

               },

               "DimeNetDiabat": 
               {
                   "n_rbf": int,
                   "cutoff": float,
                   "envelope_p": int,
                   "n_spher": int,
                   "l_spher": int,
                   "atom_embed_dim": int,
                   "n_bilinear": int,
                   "activation": str,
                   "n_convolutions": int,
                   "output_keys": list,
                   "grad_keys": list,
                   "diabat_keys": list

               },

               "DimeNetDiabatDelta":
               {
                   "n_rbf": int,
                   "cutoff": float,
                   "envelope_p": int,
                   "n_spher": int,
                   "l_spher": int,
                   "atom_embed_dim": int,
                   "n_bilinear": int,
                   "activation": str,
                   "n_convolutions": int,
                   "output_keys": list,
                   "grad_keys": list,
                   "diabat_keys": list

               },

               "DimeNetDelta":
               {
                   "n_rbf": int,
                   "cutoff": float,
                   "envelope_p": int,
                   "n_spher": int,
                   "l_spher": int,
                   "atom_embed_dim": int,
                   "n_bilinear": int,
                   "activation": str,
                   "n_convolutions": int,
                   "output_keys": list,
                   "grad_keys": list,
                   "diabat_keys": list

               },

               "SchNetDiabat":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'bond_par': float,
                   'trainable_gauss': bool,
                   'box_size': np.array,
                   'dropout_rate': float
               },



}

MODEL_DICT = {
    "SchNet": SchNet,
    "SchNetDiabat": SchNetDiabat,
    "HybridGraphConv": HybridGraphConv,
    "WeightedConformers": WeightedConformers,
    "SchNetFeatures": SchNetFeatures,
    "ChemProp3D": ChemProp3D,
    "OnlyBondUpdateCP3D": OnlyBondUpdateCP3D,
    "DimeNet": DimeNet,
    "DimeNetDiabat": DimeNetDiabat,
    "DimeNetDiabatDelta": DimeNetDiabatDelta,
    "DimeNetDelta": DimeNetDelta

}


class ParameterError(Exception):
    """Raised when a hyperparameter is of incorrect type"""

    pass


def check_parameters(params_type, params):
    """Check whether the parameters correspond to the specified types

    Args:
        params (dict)
    """
    for key, val in params.items():
        if key in params_type and not isinstance(val, params_type[key]):
            raise ParameterError("%s is not %s" % (str(key), params_type[key]))

        for model in PARAMS_TYPE.keys():
            if key == "{}_params".format(model.lower()):
                check_parameters(PARAMS_TYPE[model], val)


def get_model(params, model_type="SchNet", **kwargs):
    """Create new model with the given parameters.

    Args:
        params (dict): parameters used to construct the model
        model_type (str): name of the model to be used

    Returns:
        model (nff.nn.models)
    """

    check_parameters(PARAMS_TYPE[model_type], params)
    model = MODEL_DICT[model_type](params, **kwargs)

    return model


def load_model(path, params=None, model_type=None, **kwargs):
    """Load pretrained model from the path. If no epoch is specified,
        load the best model.

    Args:
        path (str): path where the model was trained.
        params (dict, optional): Any parameters you need to instantiate
            a model before loading its state dict. This is required for DimeNet,
            in which you can't pickle the model directly.
        model_type (str, optional): name of the model to be used
    Returns:
        model
    """

    try:
        if os.path.isdir(path):
            return torch.load(os.path.join(path, "best_model"), map_location="cpu")
        elif os.path.exists(path):
            return torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError("{} was not found".format(path))
    except (FileNotFoundError, EOFError, RuntimeError):

        assert params is not None, "Must specify params if you want to load the state dict"
        assert model_type is not None, "Must specify the model type if you want to load the state dict"

        model = get_model(params, model_type=model_type, **kwargs)

        if os.path.isdir(path):
            state_dict = torch.load(os.path.join(path, "best_model.pth.tar"), map_location="cpu")
        elif os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError("{} was not found".format(path))

        model.load_state_dict(state_dict["model"])
        return model
