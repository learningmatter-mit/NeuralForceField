"""
Script to run neural Zhu Nakamura
"""

import os
import pdb

import torch
import argparse
import json
import time
from ase import Atoms
from rdkit import Chem

from nff.io.ase_ax import NeuralFF, AtomsBatch
from nff.md.zhu_nakamura.dynamics import CombinedZhuNakamura
from nff.data import Dataset

DEFAULT_PARAMS_FILE = "job_info.json"
ANGLE_MODELS = ["DimeNet", "DimeNetDiabat", "DimeNetDiabatDelta"]
PERIODICTABLE = Chem.GetPeriodicTable()


def load_params(params_file):
    """
    Get the content from params_file
    Args:
        params_file (str): name of params_file
    Returns:
        params (dict): dictionary of parameters
    """

    with open(params_file, "r") as f:
        params = json.load(f)

    return params


def make_dataset(nxyz, all_params):
    """
    Make a dataset.
    Args:
        nxyz (list): nxyz
        all_params (dict): dictionary of parameters
    Returns:
        dataset (nff.data.dataset): dataset
    """

    props = {
        'nxyz': nxyz
    }

    cutoff = all_params["ground_params"]["cutoff"]

    dataset = Dataset(props.copy(), units='kcal/mol')
    dataset.generate_neighbor_list(cutoff=cutoff,
                                   undirected=False)

    model_type = all_params.get("model_type")
    needs_angles = (model_type in ANGLE_MODELS)
    if needs_angles:
        dataset.generate_angle_list()

    # dataset.generate_topologies(bond_dic)

    return dataset


def make_trj(all_params,
             dataset):
    """
    Make an instance of the `CombinedZhuNakamura` class.
    Args:
        sampling_params (dict): sampling parameters
        traininghyperparams (dict): training hyperparameters
        nn_id (int): ID of the neural net
        dataset (nff.data.dataset): dataset
    Returns:
        zn: instance of CombinedZhuNakamura
    """

    zhu_params = all_params["zhu_params"]
    batched_params = all_params["batched_params"]
    ground_params = all_params["ground_params"]

    if "model_path" in all_params:
        weight_path = all_params["model_path"]
    else:
        weight_path = os.path.join(all_params['weightpath'],
                                   str(all_params["nnid"]))
    batched_params.update({"weight_path": weight_path})

    nxyz = torch.cat(dataset.props["nxyz"])
    atoms = Atoms(nxyz[:, 0], positions=nxyz[:, 1:])

    batched_props = {}
    for key, val in dataset.props.items():
        if type(val[0]) is torch.Tensor and len(val[0].shape) == 0:
            batched_props.update({key: val[0].reshape(-1)})
        else:
            batched_props.update({key: val[0]})

    # # set the ground state calculator
    # # must get rid of neighbor list or else it won't be re-calculated
    # # during the MD run
    # batched_props.pop('nbr_list')

    model_type = all_params.get('model_type')
    needs_angles = (model_type in ANGLE_MODELS)
    device = all_params.get('device', 'cuda')

    # this is where we probably want to set model.grad_keys = ['energy_0_grad']
    # in the model so it doesn't calculate any excited state gradients

    nff_ase = NeuralFF.from_file(
        model_path=weight_path,
        device=device,
        output_keys=["energy_0"],
        conversion="ev",
        params=all_params,
        model_type=model_type,
        needs_angles=needs_angles,
        dataset_props=batched_props
    )

    # don't calculate nacv - unecessary for ZN and requires
    # N(N+1)/2 gradients for N states instead of N
    nff_ase.model.add_nacv = False

    # get the cutoff and skin

    atomsbatch = AtomsBatch.from_atoms(atoms=atoms,
                                       props=batched_props,
                                       needs_angles=needs_angles,
                                       device=device,
                                       undirected=False,
                                       cutoff=batched_params["cutoff"],
                                       cutoff_skin=batched_params["cutoff_skin"])

    atomsbatch.set_calculator(nff_ase)
    zn = CombinedZhuNakamura(atoms=atomsbatch,
                             zhu_params=zhu_params,
                             batched_params=batched_params,
                             ground_params=ground_params,
                             props=dataset.props,
                             model_type=model_type,
                             needs_angles=needs_angles,
                             modelparams=all_params)

    return zn


def coords_to_xyz(coords):
    nxyz = []
    for dic in coords:
        directions = ['x', 'y', 'z']
        n = float(PERIODICTABLE.GetAtomicNumber(dic["element"]))
        xyz = [dic[i] for i in directions]
        nxyz.append([n, *xyz])
    return nxyz


def main():

    parser = argparse.ArgumentParser(description="Runs neural ZN")
    parser.add_argument('paramsfile', type=str, default=DEFAULT_PARAMS_FILE,
                        help="file containing all parameters")
    args = parser.parse_args()
    job_params = load_params(args.paramsfile)
    details = job_params.get("details", {})
    all_params = {**details,
                  **{key: val for key, val in job_params.items()
                     if key != "details"}}

    if "coords" in all_params:
        coords = all_params["coords"]
        nxyz = [coords_to_xyz(coords)]
    elif "nxyz" in all_params:
        nxyz = all_params["nxyz"]
    else:
        raise Exception("No coordinates found")

    print('loading models')

    dataset = make_dataset(nxyz=nxyz, all_params=all_params)

    print('running ground state + Zhu-Nakamura dynamics')

    zn = make_trj(all_params=all_params,
                  dataset=dataset)

    zn.run()


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except Exception as e:
        raise e
        print(e)
        pdb.post_mortem()
