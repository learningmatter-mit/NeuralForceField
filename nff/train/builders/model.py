"""Helper functions to create models, functions and other classes
    while checking for the validity of hyperparameters.
"""
import os
import numpy as np
import torch
from nff.nn.models.schnet import SchNet, SchNetAuTopology, AuTopology
from nff.nn.models.hybridgraph import HybridGraphConv

PARAMS_TYPE = {"SchNet":
               {
                   'n_atom_basis': int,
                   'n_filters': int,
                   'n_gaussians': int,
                   'n_convolutions': int,
                   'cutoff': float,
                   'bond_par': float,
                   'trainable_gauss': bool,
                   'box_size': np.array
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
               "AuTopology":
               {
                  "n_features": int,
                  "n_convolutions": int,
                  "conv_type": str,
                  "conv_update_layers": list,
                  "readout_hidden_nodes": list,
                  "bond_terms": list,
                  "angle_terms": list,
                  "dihedral_terms": list,
                  "improper_terms": list,
                  "pair_terms": list,
                  "output_keys": list,
                  "trainable_prior": bool
                },


               "SchNetAuTopology":
                {
                  "autopology_params": dict,
                  "schnet_params": dict,
                  "sorted_result_keys": list,
                  "grad_keys": list,
                  "sort_results": bool,
                }

               }

MODEL_DICT = {
    "SchNet": SchNet,
    "AuTopology": AuTopology,
    "SchNetAuTopology": SchNetAuTopology,
    "HybridGraphConv": HybridGraphConv,
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
            raise ParameterError(
                '%s is not %s' % (str(key), params_type[key])
            )

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


def load_model(path):
    """Load pretrained model from the path. If no epoch is specified,
        load the best model.

    Args:
        path (str): path where the model was trained.

    Returns:
        model
    """

    if os.path.isdir(path):
        return torch.load(
            os.path.join(path, 'best_model'),
            map_location='cpu'
        )
    elif os.path.exists(path):
        return torch.load(path, map_location='cpu')
    else:
        raise FileNotFoundError('{} was not found'.format(path))
