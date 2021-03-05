"""
Script to create an NFF dataset from a summary file with information about different species
and a set of pickle files with RDKit mols for the conformers of each species.
"""

import pickle
import json
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from rdkit import Chem
import logging
from datetime import datetime
import shutil

from nff.data import Dataset, concatenate_dict
from nff.utils import tqdm_enum, parse_args, fprint, read_csv, avg_distances
import copy


KEY_MAP = {"rd_mol": "nxyz",
           "boltzmannweight": "weights",
           "relativeenergy": "energy"}

# These are keys that confuse the dataset.
EXCLUDE_KEYS = ["totalconfs", "datasets", "conformerweights",
                "uncleaned_smiles", "poplowestpct"]

# these keys are for per-conformer quantities
CONF_KEYS = ["rd_mols", "bonded_nbr_list", "bond_features",
             "atom_features"]

# disable logger to avoid annoying pickle messages
logger = logging.getLogger()
logger.disabled = True


def mol_to_smiles(rd_mol):
    """
    Get the canonical SMILES from an RDKit mol.
    Args:
        rd_mol (rdkit.Chem.rdchem.Mol): rdkit Mol
    Returns:
        smiles (str): canonical smiles
    """
    smiles = Chem.MolToSmiles(rd_mol)
    new_mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(new_mol)

    return smiles


def trim_dset(dset, good_idx):
    """
    Trim a dataest based on a set of indices you want to keep.
    Args:
        dset (nff.data.dataset): NFF dataset
        good_idx (list[int]): indices that you want to keep
    Returns:
        dset (nff.data.dataset): trimmmed NFF dataset
    """
    for key, val in dset.props.items():
        # if it's a list, take element by element and put in list
        if type(val) is list:
            dset.props[key] = [val[i] for i in good_idx]
        # otherwise can take the slice all at once
        else:
            dset.props[key] = val[good_idx]
    return dset


def get_bad_smiles(dset, good_idx):
    """
    Get the SMILES whose indices are not in `good_idx`.
    Args:
        dset (nff.data.dataset): NFF dataset
        good_idx (list[int]): indices that you want to keep
    Returns:
        bad_smiles (list[str]): smiles whose indices are not in 
            `good_idx`.
    """
    bad_smiles = [smiles for i, smiles in enumerate(dset.props["smiles"])
                  if i not in good_idx]
    return bad_smiles


def filter_same_smiles(dset):
    """
    Filter out species whose conformers don't all have the same SMILES. Can happen
    because, for example, CREST simulations can be reactive. This won't happen if
    conformers are generated using RDKit.
    Args:
        dset (nff.data.dataset): NFF dataset
    Returns:
        dset (nff.data.dataset): NFF dataset trimmed for conformers that have different
            SMILES
        bad_smiles (list[str]): The SMILES strings that we're getting rid of 
    """

    good_idx = []

    for i, batch in tqdm_enum(dset):
        rd_mols = batch["rd_mols"]
        smiles_list = [mol_to_smiles(mol) for mol in rd_mols]
        unique_smiles = list(set(smiles_list))
        # only keep if there's one unique SMILES string
        if len(unique_smiles) == 1:
            good_idx.append(i)

    # must be before trimming
    bad_smiles = get_bad_smiles(dset, good_idx)

    # trim
    dset = trim_dset(dset, good_idx)

    return dset, bad_smiles


def filter_bonds_in_nbr(cutoff, dset):
    """
    Filter out conformers whose bonds are not within the cutoff distance
    that defines the neighbor list. CP3D can't use these conformers because
    there will be bonds that don't have distance features, as the two atoms are
    not within each other's cutoff. Any conformer with bonds > 5 A is probably
    not too accurate anyway.

    Args:
        cutoff (float): neighbor list cutoff
        dset (nff.data.dataset): NFF dataset
    Returns:
        dset (nff.data.dataset): NFF dataset trimmed for above criterion
            SMILES
        bad_smiles (list[str]): The SMILES strings that we're getting rid of 
    """

    good_idx = []

    for i, batch in tqdm_enum(dset):
        bond_list = batch["bonded_nbr_list"]
        nxyz = batch["nxyz"]
        # calculate the bond lengths
        bond_lens = (nxyz[:, 1:][bond_list[:, 0]] -
                     nxyz[:, 1:][bond_list[:, 1]]).norm(dim=1)
        # only valid if they're less than the cutoff
        valid = (bond_lens < cutoff).all()
        if valid:
            good_idx.append(i)

    bad_smiles = get_bad_smiles(dset, good_idx)
    dset = trim_dset(dset, good_idx)

    return dset, bad_smiles


def get_thread_dic(sample_dic, thread, num_threads):
    """
    Given a thread (i.e., an index that tells us which
    section of the total dataset we're creating and saving),
    return the section of `sample_dic` that includes SMILES
    strings in this thread.

    Args:
        sample_dic (dict): Sample of `summary_dic` that is used
            in this combined dataset. `summary_dic` contains
            information about all smiles strings we have, except
            for their conformers.
        thread (int): Index that tells us which section of the
            total dataset that we're creating and saving
        num_threads (int): Total number of sections into which
            we're splitting and saving the dataset.
    Returns:
        sample_dic (dict): `sample_dic`, but only with species 
            from the thread we're looking at.
    """

    # sort the keys so the order is reproducible
    keys = np.array(sorted(list(
        sample_dic.keys())))

    # split the keys into `num_threads` sections and take
    # the keys in the element `thread`
    split_keys = np.array_split(keys, num_threads)
    thread_keys = split_keys[thread]

    # use these keys in `sample_dic`
    sample_dic = {key: sample_dic[key]
                  for key in thread_keys}

    return sample_dic


def get_splits(sample_dic,
               csv_folder):
    """
    Figure out which split (train, val or test) each SMILES in
    `sample_dic` belongs to.

    Args:
        sample_dic (dict): Sample of `summary_dic` that is used
            in this combined dataset. `summary_dic` contains
            information about all smiles strings we have, except
            for their conformers.
        csv_folder (str): path to folder that contains the csv files
            with the test/val/train smiles.
    Returns:
        sample_dic (dict): `sample_dic`, but with each sub-dictionary
            updated to contain the split assignment of the SMILES. 
    """

    for name in ["train", "val", "test"]:
        path = os.path.join(csv_folder, f"{name}_full.csv")
        csv_dic = read_csv(path)
        for i, smiles in enumerate(csv_dic["smiles"]):
            # add any properties present in the csv
            props = {key: csv_dic[key][i] for key in csv_dic.keys()
                     if key != "smiles"}
            sample_dic[smiles].update({"split": name,
                                       **props})

    # get rid of anything that doesn't have a split labels
    keys = list(sample_dic.keys())
    for key in keys:
        if "split" not in sample_dic[key]:
            sample_dic.pop(key)

    return sample_dic


def resave_splits(csv_folder,
                  remove_smiles):
    """
    Re-save the SMILES splits accounting for the fact that not all
    species made it into this dataset.
    Args:
        csv_folder (str): path to folder that contains the csv files
            with the test/val/train smiles.
        remove_smiles (list[str]): any SMILES strings that had to be
            removed from the NFF dataset.
    Returns:
        None
    """

    split_names = ["train", "val", "test"]

    # files have the form train_smiles.csv, train_full.csv, etc.,
    # where the "smiles" files just contain the SMILES strings,
    # but the "full" files also contain properties

    suffixes = ["smiles", "full"]

    for name in split_names:
        for suffix in suffixes:
            while True:
                path = os.path.join(csv_folder, f"{name}_{suffix}.csv")
                with open(path, "r") as f:
                    lines = f.readlines()
                keep_lines = [lines[0]]

                for line in lines[1:]:
                    smiles = line.split(",")[0].strip()
                    # don't keep the line if it contains a SMILES string
                    # in `remove_smiles`
                    if smiles not in remove_smiles:
                        keep_lines.append(line)

                new_text = "".join(keep_lines)

                # re-save the new text to a temporary file
                dt = datetime.now()
                ms_time = int(float(dt.strftime("%Y%m%d%H%M%S.%f")) * 1e3)
                tmp_path = f"{ms_time}.csv"

                with open(tmp_path, "w") as f:
                    f.write(new_text)

                # keep looping until you're sure that the file you
                # loaded and modified hasn't been changed by another
                # process while you were working

                with open(path, "r") as f:
                    new_lines = f.readlines()

                if new_lines == lines:
                    shutil.move(tmp_path, path)
                    break

                os.remove(tmp_path)


def get_sample(summary_dic,
               csv_folder,
               thread=None,
               num_threads=None):
    """
    Get the sample of `summary_dic` that is annotated with the
    test/train splits of this dataset, and only the SMILES relevant
    to this thread (i.e., this chunk of the dataset that we're
    currently working on).
    Args:
        summary_dic (dict): dictionary of the form {smiles: sub_dic},
            where `sub_dic` is a dictionary with all the species properties
            apart from its conformers.
        csv_folder (str): path to folder that contains the csv files
            with the test/val/train smiles.
        thread (int, optional): Index that tells us which section of the
            total dataset that we're creating and saving
        num_threads (int, optional): Total number of sections into which
            we're splitting and saving the dataset.
    Returns:
        sample_dic (dict): The sample of `summary_dic`.
    """

    sample_dic = copy.deepcopy(summary_dic)

    # generate train/val/test labels

    sample_dic = get_splits(sample_dic=sample_dic,
                            csv_folder=csv_folder)

    # restrict to just this thread, if we're using threads

    if thread is not None:
        sample_dic = get_thread_dic(sample_dic=sample_dic,
                                    thread=thread,
                                    num_threads=num_threads)

    return sample_dic


def load_data_from_pickle(sample_dic, pickle_folder):
    """
    Load conformer data from pickle files for this chunk 
    of the dataset.
    Args:
        sample_dic (dict): Sample of `summary_dic` that is used
            in this combined dataset. `summary_dic` contains
            information about all smiles strings we have, except
            for their conformers.
        pickle_folder (str): path to folder that contains all
            the pickle files. Each sub-dictionary in `sample_dic`
            will have the key `pickle_path`. Joining `pickle_folder`
            with `pickle_path` gives the full path to the file.
    Returns:
        overall_dic (dict): Dictionary that contains the contents of
            the pickle file for each SMILES.
    """

    overall_dic = {}
    keys = list(sample_dic.keys())

    for smiles in tqdm(keys):
        sub_dic = sample_dic[smiles]

        pickle_path = sub_dic["pickle_path"]
        full_path = os.path.join(pickle_folder, pickle_path)

        # initialize from `sample_dic`, as it may have
        # loaded some extra props from the csvs. Ignore
        # the split label as it's unnecessary.

        dic = {key: val for key, val in sub_dic.items() if
               key != "split"}

        with open(full_path, "rb") as f:
            dic.update(pickle.load(f))

        overall_dic.update({smiles: dic})

    return overall_dic


def map_key(key):
    """
    Args:
        key (str): key being used
    Returns:
        If a key is in `KEY_MAP`, returns the value specified in that dictionary.
            Otherwise just returns the key.

    """
    if key in KEY_MAP:
        return KEY_MAP[key]
    else:
        return key


def fix_iters(spec_dic, actual_confs):
    """
    Anything that is a per-species quantity will have to
    get repeated once for each of the conformers in that species
    when we make the dataset. Anything in `EXCLUDE_KEYS` shouldn't
    be included because it messes up the dataset (e.g. variable length
    strings, quantities that don't have either length 1 or length of
    the number of conformers, etc.)
    Args:
        spec_dic (dict): a dictionary of quantities associated with a 
            species.
        actual_confs (int): the number of conformers being used for this
            species. This is not the same as the total number of conformers,
            because we may have set a limit on the maximum conformers per
            species.
    Returns:
        new_spec_dic (dict): `spec_dic` updated with the above changes.

    """

    new_spec_dic = {}
    for key, val in spec_dic.items():
        if key in EXCLUDE_KEYS:
            continue
        elif type(val) in [int, float, str]:
            new_spec_dic[key] = [val] * actual_confs
        else:
            new_spec_dic[key] = val

    return new_spec_dic


def get_sorted_idx(sub_dic):
    """
    Get the indices of each conformer ordered by ascending statistical weight.
    Args:
        sub_dic (dict): dictionary for a species
    Returns:
        sorted_idx (list): Sorted indices
    """

    confs = sub_dic["conformers"]
    weight_list = []
    for i, conf in enumerate(confs):
        weight_list.append([i, conf["boltzmannweight"]])
    sorted_tuples = sorted(weight_list, key=lambda x: -x[-1])
    sorted_idx = [i[0] for i in sorted_tuples]

    return sorted_idx


def get_xyz(rd_mol):
    """
    Convert an RDKit mol to an xyz (atomic number + coordinates).
    Args:
        rd_mol (rdkit.Chem.rdchem.Mol): RDKit mol
    Returns:
        xyz (list): atomic number + coordinates 
    """

    atoms = rd_mol.GetAtoms()

    atom_nums = []
    for atom in atoms:
        atom_nums.append(atom.GetAtomicNum())

    # each conformer is a separate rdkit mol object, so each
    # mol has only one conformer

    rd_conf = rd_mol.GetConformers()[0]
    positions = rd_conf.GetPositions()

    xyz = []
    for atom_num, position in zip(atom_nums, positions):
        xyz.append([atom_num, *position])

    return xyz


def renorm_weights(spec_dic):
    """
    Renormalize weights to sum to 1, accounting for the fact that
    not using all conformers may make their sum < 1.
    Args:
        spec_dic (dict): a dictionary of quantities associated with a 
            species.
    Returns:
        spec_dic (dict): Updated `spec_dic` with renormalized weights
    """

    new_weights = np.array(spec_dic["weights"]) / sum(spec_dic["weights"])
    spec_dic["weights"] = new_weights.tolist()

    return spec_dic


def convert_data(overall_dic, max_confs):
    """
    Args:
        overall_dic (dict): Dictionary that contains the contents of
            the pickle file for each SMILES.
        max_confs (int): Maximum number of conformers per species
    Returns:
        spec_dics (list[dict]): a dictionary with data for each species
    """

    spec_dics = []
    if max_confs is None:
        max_confs = float("inf")

    for smiles in tqdm(overall_dic.keys()):

        # get everything in the dictionary except the conformer info
        sub_dic = overall_dic[smiles]
        spec_dic = {map_key(key): val for key, val in sub_dic.items()
                    if key != "conformers"}
        # must apply `str()` because the `smiles` actually has type
        # `numpy._str`
        spec_dic["smiles"] = str(smiles)

        # how many conformers we're actually using for this species
        actual_confs = min(max_confs, len(sub_dic["conformers"]))

        # fix various issues with the data
        spec_dic = fix_iters(spec_dic, actual_confs)

        # make a key and empty list for every key in the conformer
        # list
        spec_dic.update({map_key(key): [] for key
                         in sub_dic["conformers"][0].keys()
                         if key not in EXCLUDE_KEYS})

        # conformers not always ordered by weight - get the ordered
        # indices

        sorted_idx = get_sorted_idx(sub_dic)
        confs = sub_dic["conformers"]
        spec_dic["rd_mols"] = []

        # Go through the conformers from highest to lowest weight

        for idx in sorted_idx[:actual_confs]:
            conf = confs[idx]
            for conf_key in conf.keys():

                # add the RDKit mol and nxyz to the dataset
                if conf_key == "rd_mol":
                    nxyz = get_xyz(conf[conf_key])
                    spec_dic["nxyz"].append(nxyz)
                    spec_dic["rd_mols"].append(conf[conf_key])

                # add other quantities associated with the conformer
                # (e.g. Boltzmann weights)
                else:
                    new_key = map_key(conf_key)
                    if new_key not in spec_dic:
                        continue
                    spec_dic[new_key].append(conf[conf_key])

        # renormalize the weights accounting for missing conformers
        spec_dic = renorm_weights(spec_dic)
        spec_dics.append(spec_dic)

    return spec_dics


def add_missing(props_list):
    """
    There are certain quantities that are given for one species but not
    for another (e.g. whether it binds a certain protein). All quantities
    that are present for at least one species should be present in all others,
    and if not known it should be assigned as None or nan.
    Args:
        props_list (list[dict]): list of dictionaries of properties for each species
    Returns:
        props_list (list[dict]): `props_list` updated as described above
    """

    key_list = [list(props.keys()) for props in props_list]
    # dictionary of the props that have each set of keys
    key_dic = {}
    for i, keys in enumerate(key_list):
        for key in keys:
            if key not in key_dic:
                key_dic[key] = []
            key_dic[key].append(i)

    # all the possible keys
    all_keys = []
    for keys in key_list:
        all_keys += keys
    all_keys = list(set(all_keys))

    # dictionary of which props dicts are missing certain keys

    missing_dic = {}
    prop_idx = list(range(len(props_list)))
    for key in all_keys:
        missing_dic[key] = [i for i in prop_idx if
                            i not in key_dic[key]]

    for key, missing_idx in missing_dic.items():
        for i in missing_idx:

            props = props_list[i]
            given_idx = key_dic[key][0]
            given_props = props_list[given_idx]
            given_val = given_props[key]

            # If it's a list give it None
            if type(given_val) is list:
                props[key] = [None]

            # If it's a tensor give it nan
            elif type(given_val) is torch.Tensor:
                props[key] = torch.Tensor([np.nan])
                # in this case we need to change the
                # other props to have type float
                for good_idx in key_dic[key]:
                    other_props = props_list[good_idx]
                    other_props[key] = other_props[key].to(torch.float)
                    props_list[good_idx] = other_props

            props_list[i] = props

    return props_list


def clean_up_dset(dset,
                  nbrlist_cutoff,
                  strict_conformers,
                  csv_folder,
                  add_directed_idx,
                  num_procs):
    """
    Do various things to clean up the dataset after you've made it.
    Args:
        dset (nff.data.dataset): NFF dataset
        nbrlist_cutoff (float): Cutoff for two atoms to be considered
            neighbors.
        strict_conformers (bool): Whether to exclude any species whose
            conformers don't all have the same SMILES.
        csv_folder (str): path to folder that contains the csv files
            with the test/val/train smiles.
        add_directed_idx (bool): whether to calculate and add the kj
            and ji indices. These indices tell you which edges connect
            to other edges.
        num_procs (int): how many parallel threads to use when making the 
            kj and ji indices.
    Returns:
        dset (nff.data.dataset): cleaned up dataset

    """

    old_num = len(dset)

    # smiles we're getting rid of
    remove_smiles = []
    total = 3 + int(add_directed_idx)

    with tqdm(total=total) as pbar:

        # if requested, get rid of any species whose conformers have different
        # SMILES strings
        if strict_conformers:
            dset, removed = filter_same_smiles(dset)
            remove_smiles += removed

        # iterate the tqdm progress bar
        pbar.update(1)

        # Get rid of any conformers whose bond lists aren't subsets of the
        # neighbor list
        dset, removed = filter_bonds_in_nbr(nbrlist_cutoff, dset)
        remove_smiles += removed
        pbar.update(1)

        # Add the indices of the neighbor list that correspond to
        # bonded atoms. Only use one process to avoid running
        # out of memory

        dset.generate_bond_idx(num_procs=1)
        pbar.update(1)

        # Make sure the dataset is directed
        dset.make_all_directed()

        # add the kj and ji idx if requested
        if add_directed_idx:
            # only use one process to avoid running out of memory
            dset.generate_kj_ji(num_procs=1)
            pbar.update(1)

    # Re-save the train/val/test splits accounting for the fact that some
    # species are no longer there

    resave_splits(csv_folder=csv_folder,
                  remove_smiles=remove_smiles)
    new_num = old_num - len(remove_smiles)

    changed_num = old_num != new_num

    # Print a warning if the total number of species has changed
    if changed_num:
        msg = ("WARNING: the original SMILES splits have been re-saved with "
               f"{new_num} species, reduced from the original {old_num}, "
               f"because only {new_num} species made it into the final "
               "dataset. This could be because of conformers with bond "
               "lengths greater than the cutoff distance of %.2f"
               ) % nbrlist_cutoff

        if strict_conformers:
            msg += (", or because the conformers of certain species didn't "
                    "all have the same SMILES string")
        msg += "."

        fprint(msg)

    return dset


def add_features(dset,
                 extra_features,
                 parallel_feat_threads):
    """
    Add any requested features to the dataset
    Args:
        dset (nff.data.dataset): NFF dataset
        extra_features (list[dict]): list of extra features,
            where each item is a dictionary of the form
            {"name": name, "params": {params needed}}.
        parallel_feat_threads (int): how many parallel threads
            to use when making the efeatures.
    Returns:
        dset (nff.data.dataset): updated NFF dataset
    """

    for dic in tqdm(extra_features):

        name = dic["name"]
        params = dic["params"]

        if name.lower() == "e3fp":
            length = params["length"]
            fprint(f"Adding E3FP fingerprints of size {length}...")
            dset.add_e3fp(length, num_procs=parallel_feat_threads)
        if name.lower() == "whim":
            fprint("Adding whim fingerprints...")
            dset.featurize_rdkit('whim')
        if name.lower() == "morgan":
            length = params["length"]
            fprint(f"Adding Morgan fingerprints of size {length}...")
            dset.add_morgan(length)

    return dset


def make_big_dataset(spec_dics,
                     nbrlist_cutoff,
                     parallel_feat_threads):

    props_list = []
    nbr_list = []
    rd_mols_list = []

    for j, spec_dic in tqdm_enum(spec_dics):

        # Exclude keys related to individual conformers. These
        # include conformer features, in case you've already put
        # those in your pickle files. If not we'll generate them
        # below

        small_spec_dic = {key: val for key, val in spec_dic.items()
                          if key not in CONF_KEYS}

        # Treat each species' data like a regular dataset
        # and use it to generate neighbor lists

        dataset = Dataset(small_spec_dic, units='kcal/mol')

        # number of atoms in the molecule
        mol_size = len(dataset.props["nxyz"][0])
        dataset.generate_neighbor_list(cutoff=nbrlist_cutoff,
                                       undirected=False)

        # now combine the neighbor lists so that this set
        # of nxyz's can be treated like one big molecule

        nbrs = dataset.props['nbr_list']
        new_nbrs = []

        # shift by i * mol_size for each conformer
        for i in range(len(nbrs)):
            new_nbrs.append(nbrs[i] + i * mol_size)

        # add to list of conglomerated neighbor lists
        nbr_list.append(torch.cat(new_nbrs))
        dataset.props.pop('nbr_list')

        # concatenate the nxyz's
        nxyz = np.concatenate([np.array(item) for item in spec_dic["nxyz"]]
                              ).reshape(-1, 4).tolist()

        # add properties as necessary
        new_dic = {"mol_size": mol_size,
                   "nxyz": nxyz,
                   "weights": torch.Tensor(spec_dic["weights"]
                                           ).reshape(-1, 1) / sum(
                       spec_dic["weights"]),
                   "degeneracy": torch.Tensor(spec_dic["degeneracy"]
                                              ).reshape(-1, 1),
                   "energy": torch.Tensor(spec_dic["energy"]
                                          ).reshape(-1, 1),
                   "num_atoms": [len(nxyz)]}

        new_dic.update(
            {
                key: val[:1] for key, val in dataset.props.items()
                if key not in new_dic.keys()
            }
        )

        props_list.append(new_dic)
        rd_mols_list.append(spec_dic["rd_mols"])

    # Add props that are in some datasets but not others
    props_list = add_missing(props_list)
    # convert the list of dicationaries into a dicationary of lists / tensors
    props_dic = concatenate_dict(*props_list)
    # make a combined dataset where the species look like they're
    # one big molecule
    big_dataset = Dataset(props_dic, units='kcal/mol')
    # give it the proper neighbor list and rdkit mols
    big_dataset.props['nbr_list'] = nbr_list
    big_dataset.props["rd_mols"] = rd_mols_list

    # generate atom and bond features
    big_dataset.featurize(num_procs=parallel_feat_threads)

    return big_dataset


def make_avg_dataset(spec_dics,
                     nbrlist_cutoff,
                     parallel_feat_threads,
                     strict_conformers):

    if not strict_conformers:
        raise NotImplementedError

    props_list = []

    for j, spec_dic in tqdm_enum(spec_dics):

        # Exclude keys related to individual conformers. These
        # include conformer features, in case you've already put
        # those in your pickle files. If not we'll generate them
        # below

        small_spec_dic = {key: val for key, val in spec_dic.items()
                          if key not in CONF_KEYS}

        # Treat each species' data like a regular dataset
        # and use it to generate neighbor lists

        dataset = Dataset(small_spec_dic, units='kcal/mol')
        dataset.generate_neighbor_list(cutoff=nbrlist_cutoff,
                                       undirected=False)

        all_nbrs, avg_d = avg_distances(dataset)

        these_props = {"nbr_list": all_nbrs,
                       "distances": [avg_d],
                       "rd_mols": spec_dic["rd_mols"][0],
                       # we won't use the nxyz but it needs
                       # to be in an NFF dataset
                       # so we'll just use the first one
                       "nxyz": spec_dic["nxyz"][0]}

        exclude = ["weights", "degeneracy", "energy", "num_atoms",
                   "nbr_list", "distances", "rd_mols",
                   "nxyz",  *EXCLUDE_KEYS, *CONF_KEYS]

        for key, val in dataset.props.items():
            if key in exclude:
                continue

            per_conf = ((isinstance(val, list) or
                         isinstance(val, torch.Tensor))
                        and len(val) != 1)
            if per_conf:
                val = val[1]
            these_props[key] = val

        these_props.update({"num_atoms": len(spec_dic["nxyz"][0]),
                            "mol_size": len(spec_dic["nxyz"][0]),
                            "weights": torch.Tensor([1])})

        props_list.append(these_props)

    # Add props that are in some datasets but not others
    props_list = add_missing(props_list)
    # convert the list of dicationaries into a dicationary of lists / tensors
    props_dic = concatenate_dict(*props_list)

    rd_mols = copy.deepcopy(props_dic["rd_mols"])
    props_dic.pop("rd_mols")

    # make a combined dataset where the species look like they're
    # one big molecule
    final_dataset = Dataset(props_dic, units='kcal/mol')
    # generate atom and bond features
    final_dataset.props["rd_mols"] = [[i] for i in rd_mols]
    final_dataset.featurize(num_procs=parallel_feat_threads)

    return final_dataset


def make_nff_dataset(spec_dics,
                     nbrlist_cutoff,
                     parallel_feat_threads,
                     strict_conformers,
                     csv_folder,
                     extra_features,
                     add_directed_idx,
                     average_nbrs=False):
    """
    Make an NFF dataset
    Args:
        spec_dics (list[dict]): a dictionary with data for each species
        nbr_list_cutoff (float): Cutoff for two atoms to be considered
            neighbors.
        parallel_feat_threads (int): how many parallel threads
            to use when making the efeatures.
        strict_conformers (bool): Whether to exclude any species whose
            conformers don't all have the same SMILES.
        csv_folder (str): path to folder that contains the csv files
            with the test/val/train smiles.
        extra_features (list[dict]): list of extra features dictionaries
        add_directed_idx (bool): whether to calculate and add the kj
            and ji indices. These indices tell you which edges connect
            to other edges.
    Returns:
        big_dataset (nff.data.dataset): NFF dataset

    """

    fprint("Making dataset with %d species" % (len(spec_dics)))

    if average_nbrs:
        big_dataset = make_avg_dataset(spec_dics=spec_dics,
                                       nbrlist_cutoff=nbrlist_cutoff,
                                       parallel_feat_threads=parallel_feat_threads,
                                       strict_conformers=strict_conformers)
    else:
        big_dataset = make_big_dataset(spec_dics=spec_dics,
                                       nbrlist_cutoff=nbrlist_cutoff,
                                       parallel_feat_threads=parallel_feat_threads)

    # clean up
    fprint("Cleaning up dataset...")
    big_dataset = clean_up_dset(dset=big_dataset,
                                nbrlist_cutoff=nbrlist_cutoff,
                                strict_conformers=strict_conformers,
                                csv_folder=csv_folder,
                                add_directed_idx=add_directed_idx,
                                num_procs=parallel_feat_threads)

    # add any other requested features
    big_dataset = add_features(dset=big_dataset,
                               extra_features=extra_features,
                               parallel_feat_threads=parallel_feat_threads)

    return big_dataset


def get_data_folder(dataset_folder, thread):
    """
    Get the folder in which you'll save the dataset.
    Args:
        dataset_folder (str): base folder for the datasets
        thread (int): thread for chunk of dataset
    Returns:
        new_path (str): folder in which you'll save the dataset
    """

    # if we're not doing any chunks then just save in the base fodler
    if thread is None:
        return dataset_folder

    # otherwise save in base_folder/<thread>
    new_path = os.path.join(dataset_folder, str(thread))
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    return new_path


def split_dataset(dataset, idx):
    """
    Similar to `trim_dset`, but making a new dataset without modifying
    the original.
    Args:
        dataset (nff.data.dataset): NFF dataset
        idx (list[int]): indices to keep
    Returns:
        new_dataset (nff.data.dataset): new dataset with only
            `idx` indices, without modifying the old dataset.
    """

    # create a reference dataset with the right units and dummy
    # properties
    ref_props = {"nxyz": dataset.props["nxyz"][:1]}
    new_dataset = Dataset(ref_props, units=dataset.units)

    # update the properties using `dataset` and `idx`
    for key, val in dataset.props.items():
        if type(val) is list:
            new_dataset.props[key] = [val[i] for i in idx]
        else:
            new_dataset.props[key] = val[idx]

    return new_dataset


def save_splits(dataset,
                dataset_folder,
                thread,
                sample_dic):
    """
    Save the train/val/test splits of the dataset
    Args:
        dataset (nff.data.dataset): NFF dataset
        dataset_folder (str): base folder for the datasets
        thread (int): thread for chunk of dataset
        sample_dic (dict): Sample of `summary_dic` that is used
            in this combined dataset. `summary_dic` contains
            information about all smiles strings we have, except
            for their conformers.
    Returns:
        None
    """

    split_names = ["train", "val", "test"]
    split_idx = {name: [] for name in split_names}

    for i, smiles in enumerate(dataset.props['smiles']):
        split_name = sample_dic[smiles]["split"]
        split_idx[split_name].append(i)

    fprint("Saving...")

    data_folder = get_data_folder(dataset_folder, thread)

    for name in split_names:
        dset = split_dataset(dataset, split_idx[name])
        dset_path = os.path.join(data_folder, name + ".pth.tar")
        dset.save(dset_path)


def main(max_confs,
         summary_path,
         dataset_folder,
         pickle_folder,
         num_threads,
         thread,
         nbrlist_cutoff,
         csv_folder,
         parallel_feat_threads,
         strict_conformers,
         extra_features,
         add_directed_idx,
         average_nbrs,
         **kwargs):
    """
    Sample species, load their pickles, create an NFF dataset, and
    save train/val/test splits.
    Args:
        max_confs (int): Maximum number of conformers per species
        summary_path (str): Path to file with summary dictionary
        dataset_folder (str): base folder for the datasets
        pickle_folder (str): path to folder that contains all
            the pickle files. Each sub-dictionary in `sample_dic`
            will have the key `pickle_path`. Joining `pickle_folder`
            with `pickle_path` gives the full path to the file.
        num_threads (int): Total number of sections into which
            we're splitting and saving the dataset.
        thread (int): Index that tells us which section of the
            total dataset that we're creating and saving
        nbrlist_cutoff (float): Cutoff for two atoms to be considered
            neighbors.
        csv_folder (str): path to folder that contains the csv files
            with the test/val/train smiles.
        parallel_feat_threads (int): how many parallel threads
            to use when making the efeatures.
        strict_conformers (bool): Whether to exclude any species whose
            conformers don't all have the same SMILES.
        extra_features (list[dict]): list of extra features,
            where each item is a dictionary of the form
            {"name": name, "params": {params needed}}.
        add_directed_idx (bool): whether to calculate and add the kj
            and ji indices. These indices tell you which edges connect
            to other edges.
    Returns:
        None

    """

    with open(summary_path, "r") as f:
        summary_dic = json.load(f)

    fprint("Loading splits...")

    sample_dic = get_sample(summary_dic=summary_dic,
                            thread=thread,
                            num_threads=num_threads,
                            csv_folder=csv_folder)

    fprint("Loading data from pickle files...")
    overall_dic = load_data_from_pickle(sample_dic, pickle_folder)

    fprint("Converting data...")
    spec_dics = convert_data(overall_dic, max_confs)

    fprint("Combining to make NFF dataset...")
    dataset = make_nff_dataset(spec_dics=spec_dics,
                               nbrlist_cutoff=nbrlist_cutoff,
                               parallel_feat_threads=parallel_feat_threads,
                               strict_conformers=strict_conformers,
                               csv_folder=csv_folder,
                               extra_features=extra_features,
                               add_directed_idx=add_directed_idx,
                               average_nbrs=average_nbrs)

    fprint("Creating test/train/val splits...")
    save_splits(dataset=dataset,
                dataset_folder=dataset_folder,
                thread=thread,
                sample_dic=sample_dic)

    fprint((f"Complete! Saved section {thread} of the dataset in "
            f"{os.path.join(dataset_folder, str(thread))}.\n\n"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_confs', type=int, default=None,
                        help=("Maximum number of conformers to allow in any "
                              "species in your dataset. No limit if "
                              "max_confs isn't specified."))

    parser.add_argument('--nbrlist_cutoff', type=float, default=5,
                        help=("Cutoff for 3D neighbor list"))

    parser.add_argument('--summary_path', type=str)
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--pickle_folder', type=str)
    parser.add_argument('--num_threads', type=int, default=None)
    parser.add_argument('--thread', type=int, default=None)
    parser.add_argument('--prop', type=str, default=None,
                        help=("Name of property for which to generate "
                              "a proportional classification sample"))
    parser.add_argument('--csv_folder', type=str,
                        help=("Name of the folder in which "
                              "you want to save the SMILES "
                              "splits"))
    parser.add_argument('--parallel_feat_threads', type=int,
                        default=5,
                        help=("Number of parallel threads to use "
                              "when generating features"))
    parser.add_argument('--strict_conformers', action='store_true',
                        help=("Exclude any species whose conformers don't "
                              "all have the same SMILES."))
    parser.add_argument('--extra_features', type=str, default=None,
                        help=("List of dictionaries of extra features, "
                              "where each dictionary has the name of the"
                              "feature and any associated parameters. "
                              "If using the command line, "
                              "please provide as a JSON string."))
    parser.add_argument('--add_directed_idx', action='store_true',
                        help=("Add the kj and ji indices mapping out edges "
                              "that are neighbors of other edges. This takes "
                              "a fair bit of extra time, but if you're "
                              "training a ChemProp3D model, which uses edge "
                              "updates, this will save you a lot of time "
                              "during training."))

    parser.add_argument('--average_nbrs', action='store_true',
                        help=("Use one effective structure with interatomic distances "
                              "averaged over conformers"))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)

    if type(args.extra_features) == str:
        args.extra_features = json.loads(args.extra_features)

    main(**args.__dict__)
