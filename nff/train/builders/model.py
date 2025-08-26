"""Helper functions to create models, functions and other classes
while checking for the validity of hyperparameters.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from nff.nn.models.chgnet import CHGNetNFF
from nff.nn.models.conformers import WeightedConformers
from nff.nn.models.cp3d import ChemProp3D, OnlyBondUpdateCP3D
from nff.nn.models.dimenet import (
    DimeNet,
    DimeNetDelta,
    DimeNetDiabat,
    DimeNetDiabatDelta,
)
from nff.nn.models.dispersion_models import PainnDispersion
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.mace import NffScaleMACE
from nff.nn.models.painn import (
    Painn,
    Painn_NAC_OuterProd,
    Painn_Tuple,
    Painn_VecOut,
    Painn_VecOut2,
    Painn_wCP,
    PainnAdiabat,
    PainnDiabat,
    PainnDipole,
    PainnTransformer,
)
from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.models.spooky import RealSpookyNet, SpookyNet
from nff.nn.models.spooky_painn import SpookyPainn, SpookyPainnDiabat
from nff.nn.models.torchmd_net import TorchMDNet

PARAMS_TYPE = {
    "SchNet": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "n_convolutions": int,
        "cutoff": float,
        "bond_par": float,
        "trainable_gauss": bool,
        "box_size": np.array,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
        "dropout_rate": float,
    },
    "HybridGraphConv": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "mol_n_convolutions": int,
        "mol_n_cutoff": float,
        "sys_n_convolutions": int,
        "sys_n_cutoff": float,
        "V_ex_power": int,
        "V_ex_sigma": float,
        "trainable_gauss": bool,
    },
    "WeightedConformers": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "n_convolutions": int,
        "trainable_gauss": bool,
        "dropout_rate": float,
        "readoutdict": dict,
        "mol_fp_layers": list,
    },
    "SchNetFeatures": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "n_convolutions": int,
        "cutoff": float,
        "bond_par": float,
        "trainable_gauss": bool,
        "box_size": np.array,
        "dropout_rate": float,
        "n_bond_hidden": int,
        "n_bond_features": int,
        "activation": str,
    },
    "ChemProp3D": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "n_convolutions": int,
        "cutoff": float,
        "bond_par": float,
        "trainable_gauss": bool,
        "box_size": np.array,
        "dropout_rate": float,
        "cp_input_layers": list,
        "schnet_input_layers": list,
        "output_layers": list,
        "n_bond_hidden": int,
        "activation": str,
    },
    "OnlyBondUpdateCP3D": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "n_convolutions": int,
        "cutoff": float,
        "bond_par": float,
        "trainable_gauss": bool,
        "box_size": np.array,
        "schnet_dropout": float,
        "cp_dropout": float,
        "input_layers": list,
        "output_layers": list,
        "n_bond_hidden": int,
        "activation": str,
    },
    "DimeNet": {
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
    },
    "DimeNetDiabat": {
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
        "diabat_keys": list,
    },
    "DimeNetDiabatDelta": {
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
        "diabat_keys": list,
    },
    "DimeNetDelta": {
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
        "diabat_keys": list,
    },
    "SchNetDiabat": {
        "n_atom_basis": int,
        "n_filters": int,
        "n_gaussians": int,
        "n_convolutions": int,
        "cutoff": float,
        "bond_par": float,
        "trainable_gauss": bool,
        "box_size": np.array,
        "dropout_rate": float,
    },
    "Painn": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "PainnTransformer": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
    },
    "PainnDiabat": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
        "diabat_keys": list,
    },
    "PainnAdiabat": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
    },
    "Painn_VecOut": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "output_vec_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "Painn_VecOut2": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "output_vec_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "Painn_NAC_OuterProd": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "output_vec_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "Painn_Tuple": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "output_vec_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "Painn_wCP": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "output_vec_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "TorchMDNet": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
    },
    "SpookyNet": {
        "output_keys": list,
        "grad_keys": list,
        "feat_dim": int,
        "r_cut": float,
        "gamma": float,
        "bern_k": int,
        "num_conv": int,
    },
    "SpookyPainn": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
    },
    "SpookyPainnDiabat": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
        "diabat_keys": list,
    },
    "RealSpookyNet": {
        "activation": str,
        "num_features": int,
        "num_basis_functions": int,
        "num_modules": int,
        "num_residual_electron": int,
        "num_residual_pre": int,
        "num_residual_post": int,
        "num_residual_pre_local_x": int,
        "num_residual_pre_local_s": int,
        "num_residual_pre_local_p": int,
        "num_residual_pre_local_d": int,
        "num_residual_output": int,
        "basis_functions": str,
        "exp_weighting": bool,
        "cutoff": float,
        "lr_cutoff": float,
        "use_zbl_repulsion": bool,
        "use_electrostatics": bool,
        "use_d4_dispersion": bool,
        "use_irreps": bool,
        "use_nonlinear_embedding": bool,
        "compute_d4_atomic": bool,
        "module_keep_prob": float,
        "load_from": str,
        "Zmax": int,
        "zero_init": bool,
    },
    "PainnDispersion": {
        "functional": str,
        "disp_type": str,
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
    "PainnDipole": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
        "output_vec_keys": list,
        "vector_per_atom": dict,
    },
    # MACE and CHGNet params can also be loaded from pre-trained model
    "NffScaleMACE": {
        "r_max": float,
        "num_bessel": int,
        "num_polynomial_cutoff": int,
        "max_ell": int,
        "interaction": str,
        "interaction_first": str,
        "num_interactions": int,
        "num_elements": int,
        "hidden_irreps": str,
        "MLP_irreps": str,
        "atomic_energies": np.ndarray,
        "avg_num_neighbors": float,
        "atomic_numbers": list,
        # "correlation": Union[int, List[int]],
        "atomic_inter_scale": float,
        "atomic_inter_shift": float,
        "gate": str,
        "radial_MLP": list,
        "radial_type": str,
    },
    "CHGNetNFF": {  # denote not supported by ininstance
        "atom_fea_dim": int,
        "bond_fea_dim": int,
        "angle_fea_dim": int,
        # "composition_model": Union[torch.nn.Module, str],
        "num_radial": int,
        "num_angular": int,
        "n_conv": int,
        # "atom_conv_hidden_dim": Union[Sequence[int], int],
        "update_bond": bool,
        # "bond_conv_hidden_dim": Union[Sequence[int], int],
        "update_angle": bool,
        # "angle_layer_hidden_dim": Union[Sequence[int], int],
        "conv_dropout": float,
        "read_out": str,
        "gMLP_norm": str,
        "readout_norm": str,
        # "mlp_hidden_dims": Union[Sequence[int], int],
        "mlp_first": bool,
        "is_intensive": bool,
        "non_linearity": str,
        "atom_graph_cutoff": float,
        "bond_graph_cutoff": float,
        "graph_converter_algorithm": str,
        "cutoff_coeff": int,
        "learnable_rbf": bool,
        # "device": Union[int, str],
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
    "DimeNetDelta": DimeNetDelta,
    "Painn": Painn,
    "PainnTransformer": PainnTransformer,
    "PainnDiabat": PainnDiabat,
    "PainnAdiabat": PainnAdiabat,
    "TorchMDNet": TorchMDNet,
    "SpookyNet": SpookyNet,
    "SpookyPainn": SpookyPainn,
    "SpookyPainnDiabat": SpookyPainnDiabat,
    "RealSpookyNet": RealSpookyNet,
    "PainnDispersion": PainnDispersion,
    "Painn_VecOut": Painn_VecOut,
    "Painn_VecOut2": Painn_VecOut2,
    "Painn_NAC_OuterProd": Painn_NAC_OuterProd,
    "Painn_Tuple": Painn_Tuple,
    "Painn_wCP": Painn_wCP,
    "PainnDipole": PainnDipole,
    "CHGNetNFF": CHGNetNFF,
    "NffScaleMACE": NffScaleMACE,
}

DEFAULT_KWARGS = {
    "CHGNetNFF": {
        "model": "0.3.0",
    },
    "NffScaleMACE": {
        "model": "medium",
    },
}


class ParameterError(Exception):
    """Raised when a hyperparameter is of incorrect type"""


def check_parameters(params_type, params):
    """Check whether the parameters correspond to the specified types

    Args:
        params_type (dict): dictionary with the types of the parameters
        params (dict): dictionary with the parameters
    """
    for key, val in params.items():
        if val is None:
            continue
        if key in params_type and not isinstance(val, params_type[key]):
            raise ParameterError(f"{key} is not {params_type[key]}")

        for model, value in PARAMS_TYPE.items():
            if key == f"{model.lower()}_params":
                check_parameters(value, val)


def get_model(params: dict, model_type: str = "SchNet", **kwargs):
    """Create new model with the given parameters.

    Args:
        params (dict): parameters used to construct the model
        model_type (str): name of the model to be used
        kwargs (dict): any additional arguments to pass to the model

    Returns:
        model (nff.nn.models)
    """
    check_parameters(PARAMS_TYPE[model_type], params)

    if model_type in ["CHGNetNFF", "NffScaleMACE"]:
        return MODEL_DICT[model_type](**params, **kwargs)

    return MODEL_DICT[model_type](params, **kwargs)


def load_params(param_path: str) -> tuple(dict, str):
    """Load parameters from a json file. If the parameters are nested
    in a dictionary, the function will look for the keys "details" or
    "modelparams" to find the parameters.

    Args:
        param_path (str): path to the params file

    Returns:
        tuple(dict, str): parameters and model type
    """
    with open(param_path, "r") as f:
        info = json.load(f)
    keys = ["details", "modelparams"]
    params = None
    for key in keys:
        if key in info:
            params = info[key]
            break
    if params is None:
        params = info

    model_type = params["model_type"]

    return params, model_type


def load_model(path: str, params=None, model_type=None, **kwargs) -> torch.nn.Module:
    """Load pretrained model from the path. If no epoch is specified,
    load the best model. For big pre-trained models like CHGNet and MACE,
    the model is loaded independent of the path or other params.

    Args:
        path (str): path where the model was trained.
        params (dict, optional): Any parameters you need to instantiate
                a model before loading its state dict. This is required for DimeNet,
                in which you can't pickle the model directly.
        model_type (str, optional): name of the model to be used
        kwargs (dict): any additional arguments to pass to the model
    Returns:
        torch.nn.Module: a Pytorch model
    """
    # For TL with pre-trained CHGNet and MACE, we pass no path
    if model_type in ["CHGNetNFF", "NffScaleMACE"]:
        # If path is not None, we load the model from the path (usually a
        # fine-tuned model that we want to test or use for calculations)
        if path:
            # print("loading CHGNetNFF or NffScaleMACE model from path")
            try:
                return MODEL_DICT[model_type].from_file(path, **kwargs)
            except IsADirectoryError:
                return MODEL_DICT[model_type].from_file(os.path.join(path, "best_model"), **kwargs)

        if not kwargs:
            kwargs = DEFAULT_KWARGS[model_type]

        # Both CHGNet and MACE are wrapped models have a class "load" method
        # that can be used to load the pre-trained model
        # both "" and None should evaluate to False
        print(f"Loading {model_type} with kwargs {kwargs}")
        return MODEL_DICT[model_type].load(**kwargs)

    try:
        if os.path.isdir(path):
            model = torch.load(os.path.join(path, "best_model"), map_location="cpu")
        elif os.path.exists(path):
            model = torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError(f"{path} was not found")
    except (FileNotFoundError, EOFError, RuntimeError):
        param_path = os.path.join(path, "params.json")
        if os.path.isfile(param_path):
            params, model_type = load_params(param_path)

        assert params is not None, "Must specify params if you want to load the state dict"
        assert model_type is not None, "Must specify the model type if you want to load the state dict"

        model = get_model(params, model_type=model_type, **kwargs)

        if os.path.isdir(path):
            state_dict = torch.load(os.path.join(path, "best_model.pth.tar"), map_location="cpu")
        elif os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError(f"{path} was not found")  # noqa: B904

        model.load_state_dict(state_dict["model"], strict=False)

    return model
