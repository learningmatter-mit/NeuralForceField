"""
Link between Tully surface hopping and both NFF models
and JSON parameter files.
"""

import json
import os

from typing import *

import torch
from torch.utils.data import DataLoader
import numpy as np

from rdkit import Chem
from ase import Atoms

from nff.train import batch_to, batch_detach
from nff.nn.utils import single_spec_nbrs
from nff.data import Dataset, collate_dicts
from nff.utils import constants as const
from nff.utils.scatter import compute_grad
from nff.io.ase_ax import NeuralFF, AtomsBatch

PERIODICTABLE = Chem.GetPeriodicTable()
ANGLE_MODELS = ["DimeNet", "DimeNetDiabat", "DimeNetDiabatDelta"]


def make_loader(nxyz, nbr_list, num_atoms, needs_nbrs, cutoff, cutoff_skin, device, batch_size):
    props = {"nxyz": [torch.Tensor(i) for i in nxyz]}

    dataset = Dataset(props=props, units="kcal/mol", check_props=True)

    if needs_nbrs or nbr_list is None:
        nbrs = single_spec_nbrs(dset=dataset, cutoff=(cutoff + cutoff_skin), device=device, directed=True)
        dataset.props["nbr_list"] = nbrs
    else:
        dataset.props["nbr_list"] = nbr_list

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_dicts)

    return loader


def run_models(models: List, batch, device: Union[str, int]):
    """
    Gets a list of models, which contains X models that
    collectively predict Energies/Forces, NACVs, SOCs

    Args:
        models (list): list of torch models
        batch: torch batch to do inference for
        device: device on which all tensors are located
    """

    batch = batch_to(batch, device)

    results = {}
    for model in models:
        result = model(batch, inference=True)
        result = batch_detach(result)

        # merge dictionaries
        for key in result.keys():
            results[key] = result[key]

    return results


def concat_and_conv(results_list, num_atoms):
    """
    Concatenate results from separate batches and convert
    to atomic units
    """
    keys = results_list[0].keys()

    all_results = {}
    conv = const.KCAL_TO_AU

    grad_shape = [-1, num_atoms, 3]

    for key in keys:
        val = torch.cat([i[key] for i in results_list])

        if "energy_grad" in key or "force_nacv" in key:
            val *= conv["energy"] * conv["_grad"]
            val = val.reshape(*grad_shape)
        elif "energy" in key:
            val *= conv["energy"]
        elif ("nacv" in key or "NACV" in key) and "grad" in key:
            val *= conv["_grad"]
            val = val.reshape(*grad_shape)
        elif "NACP" in key and "grad" in key:
            val *= conv["_grad"]
            val = val.reshape(*grad_shape)
        elif "soc" in key or "SOC" in key:
            val *= 0.0000045563353  # cm-1 to Ha
        # else:
        #     msg = f"{key} has no known conversion"
        #     raise NotImplementedError(msg)

        all_results[key] = val.numpy()

    return all_results


def get_results(
    models,
    nxyz,
    nbr_list,
    num_atoms,
    needs_nbrs,
    cutoff,
    cutoff_skin,
    device,
    batch_size,
):
    """
    `nxyz_list` assumed to be in Angstroms
    """

    loader = make_loader(
        nxyz=nxyz,
        nbr_list=nbr_list,
        num_atoms=num_atoms,
        needs_nbrs=needs_nbrs,
        cutoff=cutoff,
        cutoff_skin=cutoff_skin,
        device=device,
        batch_size=batch_size,
    )
    results_list = []
    for batch in loader:
        results = run_models(models=models, batch=batch, device=device)
        results_list.append(results)

    all_results = concat_and_conv(results_list=results_list, num_atoms=num_atoms)

    return all_results


def coords_to_nxyz(coords):
    nxyz = []
    for dic in coords:
        directions = ["x", "y", "z"]
        n = float(PERIODICTABLE.GetAtomicNumber(dic["element"]))
        xyz = [dic[i] for i in directions]
        nxyz.append([n, *xyz])
    return np.array(nxyz)


def load_json(file):
    with open(file, "r") as f:
        info = json.load(f)

    if "details" in info:
        details = info["details"]
    else:
        details = {}
    all_params = {key: val for key, val in info.items() if key != "details"}
    all_params.update(details)

    return all_params


def make_dataset(nxyz, ground_params):
    props = {"nxyz": [torch.Tensor(nxyz)]}

    cutoff = ground_params["cutoff"]
    cutoff_skin = ground_params["cutoff_skin"]

    dataset = Dataset(props.copy(), units="kcal/mol")
    dataset.generate_neighbor_list(cutoff=(cutoff + cutoff_skin), undirected=False)

    model_type = ground_params["model_type"]
    needs_angles = model_type in ANGLE_MODELS
    if needs_angles:
        dataset.generate_angle_list()

    return dataset, needs_angles


def get_batched_props(dataset):
    batched_props = {}
    for key, val in dataset.props.items():
        if type(val[0]) is torch.Tensor and len(val[0].shape) == 0:
            batched_props.update({key: val[0].reshape(-1)})
        else:
            batched_props.update({key: val[0]})
    return batched_props


def add_calculator(atomsbatch, model_path, model_type, device, batched_props, output_keys=["energy_0"]):
    needs_angles = model_type in ANGLE_MODELS

    nff_ase = NeuralFF.from_file(
        model_path=model_path,
        device=device,
        output_keys=output_keys,
        conversion="ev",
        params=None,
        model_type=model_type,
        needs_angles=needs_angles,
        dataset_props=batched_props,
    )

    atomsbatch.set_calculator(nff_ase)


def get_atoms(ground_params, all_params):
    coords = all_params["coords"]
    nxyz = coords_to_nxyz(coords)
    atoms = Atoms(nxyz[:, 0], positions=nxyz[:, 1:])

    dataset, needs_angles = make_dataset(nxyz=nxyz, ground_params=ground_params)
    batched_props = get_batched_props(dataset)
    device = ground_params.get("device", "cuda")

    atomsbatch = AtomsBatch.from_atoms(
        atoms=atoms,
        props=batched_props,
        needs_angles=needs_angles,
        device=device,
        undirected=False,
        cutoff_skin=ground_params["cutoff_skin"],
    )

    if "model_path" in all_params:
        model_path = all_params["model_path"]
    else:
        model_path = os.path.join(all_params["weightpath"], str(all_params["nnid"]))
    add_calculator(
        atomsbatch=atomsbatch,
        model_path=model_path,
        model_type=ground_params["model_type"],
        device=device,
        batched_props=batched_props,
        output_keys=[ground_params["energy_key"]],
    )

    return atomsbatch
