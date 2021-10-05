"""
Link between Tully surface hopping and both NFF models
and JSON parameter files.
"""
import json
import os

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


def check_hop(model,
              results,
              max_gap_hop,
              surf,
              num_states):

    # **** this won't work - assumes surf is an integer
    """
    `max_gap_hop` in a.u.
    """
    gap_keys = []
    for i in range(num_states):
        if i == surf:
            continue
        upper = max([i, surf])
        lower = min([i, surf])
        key = f'energy_{upper}_energy_{lower}_delta'

        gap_keys.append(key)

    # convert max_gap_hop to kcal
    max_conv = max_gap_hop * const.AU_TO_KCAL['energy']
    gaps = torch.cat([results[key].reshape(-1, 1)
                      for key in gap_keys], dim=-1)
    can_hop = (gaps <= max_conv).sum(-1).to(torch.bool)

    return can_hop


def split_by_hop(dic,
                 can_hop,
                 num_atoms):

    hop_dic = {}
    no_hop_dic = {}

    for key, val in dic.items():
        if any(['nacv' in key, 'grad' in key, 'nxyz' in key]):
            val = torch.split(val, num_atoms)

        hop_tensor = torch.cat([item for i, item in enumerate(val)
                                if can_hop[i]])

        no_hop_tensor = torch.cat([item for i, item in enumerate(val)
                                   if not can_hop[i]])

        hop_dic[key] = hop_tensor
        no_hop_dic[key] = no_hop_tensor

    return hop_dic, no_hop_dic


def split_all(model,
              xyz,
              max_gap_hop,
              surf,
              num_states,
              batch,
              results):

    can_hop = check_hop(model=model,
                        results=results,
                        max_gap_hop=max_gap_hop,
                        surf=surf,
                        num_states=num_states)

    num_atoms = batch['num_atoms'].tolist()
    batch['xyz'] = xyz

    hop_batch, no_hop_batch = split_by_hop(dic=batch,
                                           can_hop=can_hop,
                                           num_atoms=num_atoms)

    hop_results, no_hop_results = split_by_hop(dic=results,
                                               can_hop=can_hop,
                                               num_atoms=num_atoms)

    splits = (hop_batch, no_hop_batch, hop_results, no_hop_results)

    return splits, can_hop


def init_results(num_atoms,
                 num_states):

    en_keys = [f'energy_{i}' for i in range(num_states)]
    grad_keys = [key + "_grad" for key in en_keys]
    nacv_keys = [f"nacv_{i}{j}" for i in range(num_states)
                 for j in range(num_states) if i != j]
    force_nacv_keys = ["force_" + key for key in nacv_keys]

    num_samples = len(num_atoms)
    shapes = {"energy": [num_samples],
              "grad": [num_samples, num_atoms[0], 3]}

    key_maps = {"energy": en_keys,
                "grad": [*grad_keys, *nacv_keys, *force_nacv_keys]}

    results = {}
    for key_type, keys in key_maps.items():
        shape = shapes[key_type]
        for key in keys:
            init = torch.ones(*shape) * float('nan')
            results[key] = init


def fill_results(batch,
                 these_results,
                 results,
                 idx):

    num_atoms = batch['num_atoms'].tolist()
    grad_flags = ['_grad', 'nacv']

    for key, val in these_results.keys():
        if any([flag in key for flag in grad_flags]):
            val = torch.stack(torch.split(val, num_atoms))

        results[key][idx] = val

    return results


def combine_all(no_hop_results,
                hop_results,
                no_hop_batch,
                hop_batch,
                can_hop,
                num_states,
                batch):

    num_atoms = batch['num_atoms'].tolist()
    results = init_results(num_atoms=num_atoms,
                           num_states=num_states)

    hop_idx = can_hop.nonzero()
    no_hop_idx = torch.bitwise_not(can_hop).nonzero()

    tuples = [(no_hop_batch, no_hop_results, no_hop_idx),
              (hop_batch, hop_results, hop_idx)]

    for tup in tuples:
        batch, these_results, idx = tup
        results = fill_results(batch=batch,
                               these_results=these_results,
                               results=results,
                               idx=idx)
    return results


def grad_by_split(model,
                  hop_batch,
                  hop_results,
                  no_hop_batch,
                  no_hop_results,
                  surf):

    # add all the gradients for the hop batch and results
    model.diabatic_readout.add_all_grads(xyz=hop_batch['xyz'],
                                         results=hop_results,
                                         num_atoms=hop_batch['num_atoms'],
                                         u=hop_results['U'],
                                         add_u=False)

    # just add the state gradient for the non-hop batch / results
    key = f'energy_{surf}'
    surf_grad = compute_grad(inputs=no_hop_batch['xyz'],
                             output=no_hop_results[key])
    no_hop_results[key + '_grad'] = surf_grad

    return hop_results, no_hop_results


def add_grad(model,
             batch,
             xyz,
             results,
             max_gap_hop,
             surf,
             num_states):

    # split batches and results into those that require NACVs
    # and gradients on all states, and those that only require
    # the gradient on the current state

    splits, can_hop = split_all(model=model,
                                xyz=xyz,
                                max_gap_hop=max_gap_hop,
                                surf=surf,
                                num_states=num_states,
                                batch=batch,
                                results=results)

    (hop_batch, no_hop_batch, hop_results, no_hop_results) = splits

    # add the relevant gradients

    hop_results, no_hop_results = grad_by_split(model=model,
                                                hop_batch=hop_batch,
                                                hop_results=hop_results,
                                                no_hop_batch=no_hop_batch,
                                                no_hop_results=no_hop_results,
                                                surf=surf)

    # combine everything together

    results = combine_all(no_hop_results=no_hop_results,
                          hop_results=hop_results,
                          no_hop_batch=no_hop_batch,
                          hop_batch=hop_batch,
                          can_hop=can_hop,
                          num_states=num_states,
                          batch=batch)

    return results


# def add_active_grads(batch,
#                      results,
#                      xyz,
#                      surfs,
#                      num_states):

#     num_samples = len(batch['num_atoms'])
#     num_atoms = batch['num_atoms'][0].item()

#     new_results = {f'energy_{i}_grad':
#                    np.ones(num_samples, num_atoms, 3)
#                    * float('nan')
#                    for i in range(num_states)}


def run_model(model,
              batch,
              device,
              surf,
              max_gap_hop,
              num_states,
              all_engrads,
              nacv):
    """
    `max_gap_hop` in a.u.
    """

    batch = batch_to(batch, device)

    # # case 1: we only want one state gradient
    # separate_grads = (not nacv) and (not all_engrads)
    # if separate_grads:
    #     xyz = batch['nxyz'][:, 1:]
    #     xyz.requires_grad = True

    # # case 2: we want both state gradients but
    # # no nacv
    # # Or case 3: we want both state gradients and nacv
    # else:
    #     xyz = None

    xyz = None

    model.add_nacv = nacv
    results = model(batch,
                    xyz=xyz,
                    add_nacv=nacv,
                    # add_grad=all_engrads,
                    add_grad=True,
                    add_gap=True,
                    add_u=True,
                    inference=True)

    # If we use NACV then we can come back to what's commented
    # out below, where you only ask for gradients NACVs among states
    # close to each other

    # For now just take the gradient on the active surfaces

    # if separate_grads:
    #     results = add_active_grads()

    # if not all_grads:
    #     results = add_grad(model=model,
    #                        batch=batch,
    #                        xyz=xyz,
    #                        results=results,
    #                        max_gap_hop=max_gap_hop,
    #                        surf=surf,
    #                        num_states=num_states)

    results = batch_detach(results)

    return results


def get_phases(U, old_U):
    # Compute overlap
    S = np.einsum('...ki, ...kj -> ...ij',
                  old_U, U)

    # Take the element in each column with the
    # largest absolute value, not just the diagonal.
    # When the two diabatic states switch energy
    # orderings through a CI, the adiabatic states
    # that are most similar to each other will have
    # different orderings.

    num_states = U.shape[-1]

    max_idx = abs(S).argmax(axis=1)
    num_samples = S.shape[0]

    S_max = np.take_along_axis(
        S.transpose(0, 2, 1),
        max_idx.reshape(num_samples, num_states, 1),
        axis=2
    ).transpose(0, 2, 1)

    new_phases = np.sign(S_max)

    return new_phases


def update_phase(new_phases,
                 i,
                 j,
                 results,
                 key,
                 num_atoms):

    phase = ((new_phases[:, :, i] * new_phases[:, :, j])
             .reshape(-1, 1, 1))

    updated = np.concatenate(
        np.split(results[key], num_atoms)
    ).reshape(-1, num_atoms[0], 3) * phase

    results[key] = updated

    return results


def correct_nacv(results,
                 old_U,
                 num_atoms,
                 num_states):
    """
    Stack the non-adiabatic couplings and correct their
    phases. Also correct the phases of U.
    """

    # get phase correction
    new_phases = get_phases(U=results["U"],
                            old_U=old_U)

    new_U = results["U"] * new_phases
    results["U"] = new_U

    # Stack NACVs and multiply by new phases
    # They can be stacked because only one type of molecule
    # is used in a batched simulation

    for i in range(num_states):
        for j in range(num_states):
            if j == i:
                continue

            keys = [f"force_nacv_{i}{j}", f"nacv_{i}{j}"]

            for key in keys:
                # e.g. if no states are close enough for
                # hopping

                if key not in results:
                    continue

                results = update_phase(
                    new_phases=new_phases,
                    i=i,
                    j=j,
                    results=results,
                    key=key,
                    num_atoms=num_atoms)

    return results


def batched_calc(model,
                 batch,
                 device,
                 num_states,
                 surf,
                 max_gap_hop,
                 all_engrads,
                 nacv):
    """
    Get model results from a batch, including
    nacv phase correction
    """

    results = run_model(model=model,
                        batch=batch,
                        device=device,
                        surf=surf,
                        max_gap_hop=max_gap_hop,
                        num_states=num_states,
                        all_engrads=all_engrads,
                        nacv=nacv)

    return results


def concat_and_conv(results_list,
                    num_atoms,
                    diabat_keys):
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

        if 'energy_grad' in key or 'force_nacv' in key:
            val *= conv['energy'] * conv['_grad']
            val = val.reshape(*grad_shape)
        elif 'energy' in key or key in diabat_keys:
            val *= conv['energy']
        elif 'nacv' in key:
            val *= conv['_grad']
            val = val.reshape(*grad_shape)
        # else:
        #     msg = f"{key} has no known conversion"
        #     raise NotImplementedError(msg)

        all_results[key] = val.numpy()

    return all_results


def make_loader(nxyz,
                nbr_list,
                num_atoms,
                needs_nbrs,
                cutoff,
                cutoff_skin,
                device,
                batch_size):

    props = {"nxyz": [torch.Tensor(i)
                      for i in nxyz]}

    dataset = Dataset(props=props,
                      units='kcal/mol',
                      check_props=True)

    if needs_nbrs or nbr_list is None:
        nbrs = single_spec_nbrs(dset=dataset,
                                cutoff=(cutoff +
                                        cutoff_skin),
                                device=device,
                                directed=True)
        dataset.props['nbr_list'] = nbrs
    else:
        dataset.props['nbr_list'] = nbr_list

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=collate_dicts)

    return loader


def timing(func):
    import time

    def my_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%.2f seconds" % delta)

        return result

    return my_func


# @timing
def get_results(model,
                nxyz,
                nbr_list,
                num_atoms,
                needs_nbrs,
                cutoff,
                cutoff_skin,
                device,
                batch_size,
                old_U,
                num_states,
                surf,
                max_gap_hop,
                all_engrads,
                nacv,
                diabat_keys):
    """
    `nxyz_list` assumed to be in Angstroms
    """

    loader = make_loader(nxyz=nxyz,
                         nbr_list=nbr_list,
                         num_atoms=num_atoms,
                         needs_nbrs=needs_nbrs,
                         cutoff=cutoff,
                         cutoff_skin=cutoff_skin,
                         device=device,
                         batch_size=batch_size)
    results_list = []
    for batch in loader:
        results = batched_calc(model=model,
                               batch=batch,
                               device=device,
                               num_states=num_states,
                               surf=surf,
                               max_gap_hop=max_gap_hop,
                               all_engrads=all_engrads,
                               nacv=nacv)
        results_list.append(results)

    all_results = concat_and_conv(results_list=results_list,
                                  num_atoms=num_atoms,
                                  diabat_keys=diabat_keys)

    if old_U is not None:
        all_results = correct_nacv(results=all_results,
                                   old_U=old_U,
                                   num_atoms=[num_atoms] * old_U.shape[0],
                                   num_states=num_states)

    return all_results


def coords_to_xyz(coords):
    nxyz = []
    for dic in coords:
        directions = ['x', 'y', 'z']
        n = float(PERIODICTABLE.GetAtomicNumber(dic["element"]))
        xyz = [dic[i] for i in directions]
        nxyz.append([n, *xyz])
    return np.array(nxyz)


def load_json(file):

    with open(file, 'r') as f:
        info = json.load(f)

    if 'details' in info:
        details = info['details']
    else:
        details = {}
    all_params = {key: val for key, val in info.items()
                  if key != "details"}
    all_params.update(details)

    return all_params


def make_dataset(nxyz,
                 ground_params):
    props = {
        'nxyz': [torch.Tensor(nxyz)]
    }

    cutoff = ground_params["cutoff"]
    cutoff_skin = ground_params["cutoff_skin"]

    dataset = Dataset(props.copy(), units='kcal/mol')
    dataset.generate_neighbor_list(cutoff=(cutoff + cutoff_skin),
                                   undirected=False)

    model_type = ground_params["model_type"]
    needs_angles = (model_type in ANGLE_MODELS)
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


def add_calculator(atomsbatch,
                   model_path,
                   model_type,
                   device,
                   batched_props):

    needs_angles = (model_type in ANGLE_MODELS)

    nff_ase = NeuralFF.from_file(
        model_path=model_path,
        device=device,
        output_keys=["energy_0"],
        conversion="ev",
        params=None,
        model_type=model_type,
        needs_angles=needs_angles,
        dataset_props=batched_props
    )

    atomsbatch.set_calculator(nff_ase)


def get_atoms(ground_params,
              all_params):

    coords = all_params["coords"]
    nxyz = coords_to_xyz(coords)
    atoms = Atoms(nxyz[:, 0],
                  positions=nxyz[:, 1:])

    dataset, needs_angles = make_dataset(nxyz=nxyz,
                                         ground_params=ground_params)
    batched_props = get_batched_props(dataset)
    device = ground_params.get('device', 'cuda')

    atomsbatch = AtomsBatch.from_atoms(atoms=atoms,
                                       props=batched_props,
                                       needs_angles=needs_angles,
                                       device=device,
                                       undirected=False,
                                       cutoff_skin=ground_params['cutoff_skin'])

    if 'model_path' in all_params:
        model_path = all_params['model_path']
    else:
        model_path = os.path.join(all_params['weightpath'],
                                  str(all_params["nnid"]))
    add_calculator(atomsbatch=atomsbatch,
                   model_path=model_path,
                   model_type=ground_params["model_type"],
                   device=device,
                   batched_props=batched_props)

    return atomsbatch
