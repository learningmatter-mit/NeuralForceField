import numpy as np
from concurrent import futures
import copy
import torch
import os

from nff.data.features import (make_rd_mols, featurize_bonds,
                               featurize_atoms, BOND_FEAT_TYPES,
                               ATOM_FEAT_TYPES)

NUM_PROCS = 25


def split_dataset(dataset, num):

    datasets = []
    idx = range(len(dataset))
    splits = np.array_split(idx, num)

    for split in splits:

        min_split = split[0]
        max_split = split[-1] + 1
        new_props = {key: val[min_split: max_split] for key, val
                     in dataset.props.items()}

        new_dataset = dataset.copy()
        new_dataset.props = new_props
        datasets.append(new_dataset)

    return datasets


def rejoin_props(datasets):
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

    cpu_count = os.cpu_count()
    with futures.ProcessPoolExecutor(num_workers=cpu_count) as executor:
        future_objs = []
        for kwargs in kwargs_list:
            result = executor.submit(func, **kwargs)
            future_objs.append(result)

    result_dsets = [obj.result() for obj in future_objs]

    return result_dsets


def rd_parallel(datasets):

    kwargs_list = [{"dataset": dataset, "verbose": False}
                   for dataset in datasets]
    result_dsets = gen_parallel(func=make_rd_mols,
                                kwargs_list=kwargs_list)

    return result_dsets


def bonds_parallel(datasets, feat_types):

    kwargs_list = [{"dataset": dataset, "feat_types": feat_types}
                   for dataset in datasets]
    result_dsets = gen_parallel(func=featurize_bonds,
                                kwargs_list=kwargs_list)

    return result_dsets


def atoms_parallel(datasets, feat_types):

    kwargs_list = [{"dataset": dataset, "feat_types": feat_types}
                   for dataset in datasets]
    result_dsets = gen_parallel(func=featurize_atoms,
                                kwargs_list=kwargs_list)

    return result_dsets


def summarize_rd(new_sets, first_set):
    tried = len(first_set)
    succ = sum([len(d) for d in new_sets])
    pct = succ / tried * 100
    print("Converted %d of %d molecules (%.2f%%)." %
          (succ, tried, pct))


def featurize_parallel(dataset,
                       num_procs,
                       bond_feats=BOND_FEAT_TYPES,
                       atom_feats=ATOM_FEAT_TYPES):

    print("Featurizing dataset with {} parallel processes.".format(
        num_procs))
    datasets = split_dataset(dataset=dataset, num=num_procs)

    print("Converting xyz to RDKit mols...")
    datasets = rd_parallel(datasets)
    summarize_rd(new_sets=datasets, first_set=dataset)

    print("Featurizing bonds...")
    datasets = bonds_parallel(datasets, feat_types=bond_feats)
    print("Completed featurizing bonds.")

    print("Featurizing atoms...")
    datasets = atoms_parallel(datasets, feat_types=atom_feats)
    print("Completed featurizing atoms.")

    new_props = rejoin_props(datasets)
    dataset.props = new_props

    new_props.pop("rd_mols")
    new_props["bonded_nbr_list"] = copy.deepcopy(new_props["bond_list"])
    new_props.pop("bond_list")
