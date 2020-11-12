"""
Tools for applying functions in parallel to the dataset
"""

import numpy as np
from concurrent import futures
import copy
import torch

from nff.utils import fprint
from nff.data.features import (make_rd_mols,
                               featurize_bonds,
                               featurize_atoms,
                               add_e3fp,
                               BOND_FEAT_TYPES,
                               ATOM_FEAT_TYPES)
from nff.data.graphs import kj_ji_to_dset, add_bond_idx


NUM_PROCS = 5


def split_dataset(dataset, num):
    """
    Split a dataset into smaller chunks.
    Args:
        dataset (nff.data.dataset): NFF dataset
        num (int): number of chunks
    Returns:
        datasets (list): list of smaller datasets
    """

    datasets = []
    idx = range(len(dataset))
    splits = np.array_split(idx, num)

    for split in splits:

        if len(split) == 0:
            continue
        min_split = split[0]
        max_split = split[-1] + 1
        new_props = {key: val[min_split: max_split] for key, val
                     in dataset.props.items()}

        new_dataset = dataset.__class__(props=new_props,
                                        check_props=False,
                                        units=dataset.units)
        datasets.append(new_dataset)

    return datasets


def rejoin_props(datasets):
    """
    Rejoin properties from datasets into one dictionary of
    properties.
    Args:
        datasets (list): list of smaller datasets
    Returns:
        new_props (dict): combined properties
    """
    new_props = {}
    for dataset in datasets:
        for key, val in dataset.props.items():
            if key not in new_props:
                new_props[key] = val
                continue

            if type(val) is list:
                new_props[key] += val
            else:
                new_props[key] = torch.cat([
                    new_props[key], val], dim=0)

    return new_props


def gen_parallel(func, kwargs_list):

    if len(kwargs_list) == 1:
        kwargs = kwargs_list[0]
        kwargs["track"] = True
        return [func(**kwargs)]

    with futures.ProcessPoolExecutor() as executor:
        future_objs = []
        # go through each set of kwargs
        for i, kwargs in enumerate(kwargs_list):

            # monitor with tqdm for the first process only
            # so that they don't print on top of each other
            kwargs["track"] = (i == 0)
            result = executor.submit(func, **kwargs)

            # `future_objs` are the results of applying each function
            future_objs.append(result)

    result_dsets = [obj.result() for obj in future_objs]

    # this is unnecessary
    executor.shutdown(wait=True)

    return result_dsets


def rd_parallel(datasets, check_smiles=False):
    """
    Generate RDKit mols for the dataset in parallel.
    Args:
        datasets (list): list of smaller datasets
        check_smiles (bool): exclude any species whose
            SMILES strings aren't the same as the 
    Returns:
        results_dsets (list): list of datasets with 
            RDKit mols.
    """

    kwargs_list = [{"dataset": dataset, "verbose": False,
                    "check_smiles": check_smiles}
                   for dataset in datasets]
    result_dsets = gen_parallel(func=make_rd_mols,
                                kwargs_list=kwargs_list)

    return result_dsets


def bonds_parallel(datasets, feat_types):
    """
    Generate bond lists and bond features for the dataset 
    in parallel.
    Args:
        datasets (list): list of smaller datasets
        feat_types (list[str]): types of bond features to
            use 
    Returns:
        results_dsets (list): list of datasets with 
            bond lists and features.
    """

    kwargs_list = [{"dataset": dataset, "feat_types": feat_types}
                   for dataset in datasets]
    result_dsets = gen_parallel(func=featurize_bonds,
                                kwargs_list=kwargs_list)

    return result_dsets


def atoms_parallel(datasets, feat_types):
    """
    Generate atom features for the dataset 
    in parallel.
    Args:
        datasets (list): list of smaller datasets
        feat_types (list[str]): types of atom features to
            use 
    Returns:
        results_dsets (list): list of datasets with 
            atom features.
    """

    kwargs_list = [{"dataset": dataset, "feat_types": feat_types}
                   for dataset in datasets]
    result_dsets = gen_parallel(func=featurize_atoms,
                                kwargs_list=kwargs_list)

    return result_dsets


def e3fp_parallel(datasets, fp_length):
    """
    Generate E3FP fingerprints for each conformer in the
    dataset.
    Args:
        datasets (list): list of smaller datasets
        fp_length (int): fingerprint length
    Returns:
        results_dsets (list): list of datasets with 
            E3FP fingerprints.
    """
    kwargs_list = [{"rd_dataset": dataset, "fp_length": fp_length} for
                   dataset in datasets]

    result_dsets = gen_parallel(func=add_e3fp,
                                kwargs_list=kwargs_list)

    return result_dsets


def kj_ji_parallel(dsets):
    kwargs_list = [{"dataset": dataset} for
                   dataset in dsets]

    result_dsets = gen_parallel(func=kj_ji_to_dset,
                                kwargs_list=kwargs_list)

    return result_dsets


def bond_idx_parallel(dsets):
    kwargs_list = [{"dataset": dataset} for
                   dataset in dsets]

    result_dsets = gen_parallel(func=add_bond_idx,
                                kwargs_list=kwargs_list)

    return result_dsets


def summarize_rd(new_sets, first_set):
    """
    Summarize how many RDKit mols were successfully made.
    Args:
        first_set (nff.data.dataset): initial NFF dataset
        new_sets (list): chunks of new datasets updated
            with RDKit mols.
    Returns:
        None
    """
    tried = len(first_set)
    succ = sum([len(d) for d in new_sets])
    pct = succ / tried * 100
    fprint("Converted %d of %d molecules (%.2f%%)." %
           (succ, tried, pct))


def featurize_parallel(dataset,
                       num_procs,
                       bond_feats=BOND_FEAT_TYPES,
                       atom_feats=ATOM_FEAT_TYPES):
    """
    Add RDKit mols, atom features and bond features to a dataset in 
    parallel.
    Args:
         dataset (nff.data.dataset): NFF dataset
         num_procs (int): number of parallel processes
         bond_feats (list[str]): names of bond features
         atom_feats (list[str]): names of atom features
    Returns:
        None
    """

    # offsets can be sparse tensors which in torch version <= 1.3
    # can't yet be pickled. So we have to remove them to not cause
    # errors in the pickling during parallelization

    add_offsets = False
    if "offsets" in dataset.props:
        offsets = copy.deepcopy(dataset.props["offsets"])
        add_offsets = True
        dataset.props.pop("offsets")

    msg = f"Featurizing dataset with {num_procs} parallel processes."
    if num_procs == 1:
        msg = msg.replace("processes", "process")
    fprint(msg)

    # split the dataset so processes can act in parallel on the chunks
    datasets = split_dataset(dataset=dataset, num=num_procs)

    # add RDKit mols if they're not already in the dataset
    has_rdmols = all(['rd_mols' in dset.props for dset in datasets])
    if not has_rdmols:
        fprint("Converting xyz to RDKit mols...")
        datasets = rd_parallel(datasets)
        summarize_rd(new_sets=datasets, first_set=dataset)

    fprint("Featurizing bonds...")
    datasets = bonds_parallel(datasets, feat_types=bond_feats)

    fprint("Featurizing atoms...")
    datasets = atoms_parallel(datasets, feat_types=atom_feats)

    # rejoin the dataset
    new_props = rejoin_props(datasets)
    dataset.props = new_props

    # rename the bond list as `bonded_nbr_list`
    new_props["bonded_nbr_list"] = copy.deepcopy(new_props["bond_list"])
    new_props.pop("bond_list")

    if add_offsets:
        dataset.props["offsets"] = offsets


def add_e3fp_parallel(dataset,
                      fp_length,
                      num_procs):
    """
    Add E3FP fingerprints to a dataset in parallel.
    Args:
         dataset (nff.data.dataset): NFF dataset
         fp_length (int): fingerprint length
         num_procs (int): number of parallel processes
    Returns:
        None
    """
    msg = f"Adding E3FP fingerprints with {num_procs} parallel processes."
    if num_procs == 1:
        msg = msg.replace("processes", "process")
    fprint(msg)

    # split the dataset, run E3FP in parallel, and rejoin it
    datasets = split_dataset(dataset=dataset, num=num_procs)
    datasets = e3fp_parallel(datasets, fp_length=fp_length)
    new_props = rejoin_props(datasets)
    dataset.props = new_props


def add_kj_ji_parallel(dataset, num_procs):

    fprint((f"Adding kj and ji indices with {num_procs} "
            "parallel processes"))

    datasets = split_dataset(dataset=dataset, num=num_procs)
    datasets = kj_ji_parallel(datasets)
    new_props = rejoin_props(datasets)
    dataset.props = new_props


def add_bond_idx_parallel(dataset, num_procs):

    fprint((f"Adding bond indices with {num_procs} "
            "parallel processes"))

    datasets = split_dataset(dataset=dataset, num=num_procs)
    datasets = bond_idx_parallel(datasets)
    new_props = rejoin_props(datasets)
    dataset.props = new_props
