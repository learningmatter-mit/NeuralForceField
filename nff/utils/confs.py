"""
Tools for manipulating conformer numbers in a dataset.
"""

import torch
import math
import numpy as np
import copy

from nff.utils.misc import tqdm_enum

REINDEX_KEYS = ["nbr_list", "bonded_nbr_list"]
NBR_IDX_KEYS = ["kj_idx", "ji_idx", "bond_idx"]
PER_CONF_KEYS = ["energy"]


def assert_ordered(batch):
    """
    Make sure the conformers are ordered by weight.
    Args:
        batch (dict): dictionary of properties for one
            species.
    Returns:
        None
    """

    weights = batch["weights"].reshape(-1).tolist()
    sort_weights = sorted(weights,
                          key=lambda x: -x)
    assert weights == sort_weights


def get_batch_dic(batch,
                  idx_dic,
                  num_confs):
    """
    Get some conformer information about the batch.
    Args:
        batch (dict): Properties of the species in this batch
        idx_dic (dict): Dicationary of the form
            {smiles: indices}, where `indices` are the indices
            of the conformers you want to keep. If not specified,
            then the top `num_confs` conformers with the highest
            statistical weight will be used.
        num_confs (int): Number of conformers to keep
    Returns:
        info_dic (dict): Dictionary with extra conformer 
            information about the batch

    """

    mol_size = batch["mol_size"]
    old_num_atoms = batch["num_atoms"]

    # number of conformers in the batch is number of atoms / atoms per mol
    confs_in_batch = old_num_atoms // mol_size

    # new number of atoms after trimming
    new_num_atoms = int(mol_size * min(
        confs_in_batch, num_confs))

    if idx_dic is None:

        assert_ordered(batch)
        # new number of conformers after trimming
        real_num_confs = min(confs_in_batch, num_confs)
        conf_idx = list(range(real_num_confs))

    else:
        smiles = batch["smiles"]
        conf_idx = idx_dic[smiles]
        real_num_confs = len(conf_idx)

    info_dic = {"conf_idx": conf_idx,
                "real_num_confs": real_num_confs,
                "old_num_atoms": old_num_atoms,
                "new_num_atoms": new_num_atoms,
                "confs_in_batch": confs_in_batch,
                "mol_size": mol_size}

    return info_dic


def to_xyz_idx(batch_dic):
    """
    Get the indices of the nxyz corresponding to atoms in conformers
    we want to keep.
    Args:
        batch_dic (dict): Dictionary with extra conformer 
            information about the batch
    Returns:
        xyz_conf_all_idx (torch.LongTensor): nxyz indices of atoms
            in the conformers we want to keep.
    """

    confs_in_batch = batch_dic["confs_in_batch"]
    mol_size = batch_dic["mol_size"]
    # Indices of the conformers we're keeping (usually of the form [0, 1, ...
    # `max_confs`], unless specified otherwise)
    conf_idx = batch_dic["conf_idx"]

    # the xyz indices of where each conformer starts
    xyz_conf_start_idx = [i * mol_size for i in range(confs_in_batch + 1)]
    # a list of the full set of indices for each conformer
    xyz_conf_all_idx = []

    # go through each conformer index, get the corresponding xyz indices,
    # and append them to xyz_conf_all_idx

    for conf_num in conf_idx:

        start_idx = xyz_conf_start_idx[conf_num]
        end_idx = xyz_conf_start_idx[conf_num + 1]
        full_idx = torch.arange(start_idx, end_idx)

        xyz_conf_all_idx.append(full_idx)

    # concatenate into tensor
    xyz_conf_all_idx = torch.cat(xyz_conf_all_idx)

    return xyz_conf_all_idx


def split_nbrs(nbrs,
               mol_size,
               confs_in_batch,
               conf_idx):
    """
    Get the indices of the neighbor list that correspond to conformers 
    we're keeping.
    Args:
        nbrs (torch.LongTensor): neighbor list
        mol_size (int): Number of atoms in each conformer
        confs_in_batch (int): Total number of conformers in the batch
        conf_idx (list[int]): Indices of the conformers we're keeping 
    Returns:
        tens_idx (torch.LongTensor): nbr indices of conformers we're
            keeping.
    """

    split_idx = []
    # The cutoff of the atom indices for each conformer
    cutoffs = [i * mol_size - 1 for i in range(1, confs_in_batch + 1)]

    for i in conf_idx:

        # start index of the conformer
        start = cutoffs[i] - mol_size + 1
        # end index of the conformer
        end = cutoffs[i]

        # mask so that all elements of the neighbor list are
        # >= start and <= end
        mask = (nbrs[:, 0] <= end) * (nbrs[:, 1] <= end)
        mask *= (nbrs[:, 0] >= start) * (nbrs[:, 1] >= start)

        idx = mask.nonzero().reshape(-1)
        split_idx.append(idx)

    tens_idx = torch.cat(split_idx)

    return tens_idx


def to_nbr_idx(batch_dic, nbrs):
    """
    Apply `split_nbrs` given `batch_dic`
    Args:
        batch_dic (dict): Dictionary with extra conformer 
            information about the batch
        nbrs (torch.LongTensor): neighbor list
    Returns:
        split_nbr_idx (torch.LongTensor): nbr indices of conformers we're
            keeping. 
    """

    mol_size = batch_dic["mol_size"]
    confs_in_batch = batch_dic["confs_in_batch"]
    conf_idx = batch_dic["conf_idx"]

    split_nbr_idx = split_nbrs(nbrs=nbrs,
                               mol_size=mol_size,
                               confs_in_batch=confs_in_batch,
                               conf_idx=conf_idx)

    return split_nbr_idx


def update_weights(batch, batch_dic):
    """
    Readjust weights so they sum to 1.
    Args:
        batch_dic (dict): Dictionary with extra conformer 
            information about the batch
        batch (dict): Batch dictionary
    Returns:
        new_weights (torch.Tensor): renormalized weights
    """

    old_weights = batch["weights"]

    conf_idx = torch.LongTensor(batch_dic["conf_idx"])
    new_weights = old_weights[conf_idx]
    new_weights /= new_weights.sum()
    if torch.isnan(new_weights).any():
        new_weights = torch.ones_like(old_weights[conf_idx])
        new_weights /= new_weights.sum()
    return new_weights


def update_nbr_idx_keys(dset, batch, i, old_nbrs, num_confs):
    """
    When trimming a dataset to a certain number of conformers,
    update any quantities that depend on indices of the nbr list
    (e.g. bond_idx, kj_idx and ji_idx).
    Args:
        dset (nff.data.Dataset): nff dataset
        batch (dict): batch of the dataset
        i (int): index of the current batch in the dataset
        old_nbrs (torch.LongTensor): old neighbor list of this
            batch before trimming
        num_confs (int): number of conformers we're trimming the
            dataset to.
    Returns:
        None
    """

    # make a mask for the neighbor list indices that are being kept

    mol_size = batch['mol_size']
    for j in range(num_confs):
        mask = (old_nbrs[:, 0] < (j + 1) * mol_size
                ) * (old_nbrs[:, 0] >= j * mol_size)
        if j == 0:
            total_mask = copy.deepcopy(mask)
        else:
            total_mask += copy.deepcopy(mask)

    # go through each neighbor list index-dependent key and fix
    total_mask = total_mask.to(torch.bool)
    for key in NBR_IDX_KEYS:
        if key not in dset.props:
            continue
        # the indices that we're keeping are determined by `total_mask`.
        # Since the values of this quantity are neighbor list indices,
        # we can just select the corresponding indices of `total_mask`.
        keep_idx = total_mask[batch[key]]
        dset.props[key][i] = batch[key][keep_idx]


def update_per_conf(dataset, i, old_num_atoms, new_n_confs):
    mol_size = dataset.props["mol_size"][i]
    for key in PER_CONF_KEYS:
        if key not in dataset.props:
            continue
        val = dataset.props[key][i]
        dataset.props[key][i] = val[:new_n_confs]


def update_dset(batch, batch_dic, dataset, i):
    """
    Update the dataset with the new values for the requested
    number of conformers, for species at index i.
    Args:
        batch (dict): Batch dictionary
        batch_dic (dict): Dictionary with extra conformer 
            information about the batch
        dataset (nff.data.dataset): NFF dataset
        i (int): index of the species whose info we're updating
    Returns:
        dataset (nff.data.dataset): updated NFF dataset
    """

    bond_nbrs = batch["bonded_nbr_list"]
    nbr_list = batch["nbr_list"]
    bond_feats = batch["bond_features"]
    atom_feats = batch["atom_features"]
    nxyz = batch["nxyz"]

    # get the indices of the nyxz, nbr list, and bonded
    # nbr list that are contained within the requested number
    # of conformers

    conf_xyz_idx = to_xyz_idx(batch_dic)
    bond_nbr_idx = to_nbr_idx(batch_dic, bond_nbrs)
    all_nbr_idx = to_nbr_idx(batch_dic, nbr_list)

    # change the number of atoms to the proper value
    old_num_atoms = copy.deepcopy(dataset.props["num_atoms"][i])
    dataset.props["num_atoms"][i] = batch_dic["new_num_atoms"]
    # get the right nxyz
    dataset.props["nxyz"][i] = nxyz[conf_xyz_idx]

    # convert the neighbor lists
    dataset.props["bonded_nbr_list"][i] = bond_nbrs[bond_nbr_idx]
    dataset.props["nbr_list"][i] = nbr_list[all_nbr_idx]

    # get the atom and bond features at the right indices
    dataset.props["bond_features"][i] = bond_feats[bond_nbr_idx]
    dataset.props["atom_features"][i] = atom_feats[conf_xyz_idx]

    # renormalize weights
    dataset.props["weights"][i] = update_weights(batch,
                                                 batch_dic)

    # update anything else that's a per-conformer quantity
    update_per_conf(dataset, i, old_num_atoms, batch_dic["real_num_confs"])

    # update anything that depends on the indexing of the nbr list
    update_nbr_idx_keys(dset=dataset,
                        batch=batch,
                        i=i,
                        old_nbrs=nbr_list,
                        num_confs=batch_dic["real_num_confs"])

    return dataset


def trim_confs(dataset,
               num_confs,
               idx_dic,
               enum_func=None):
    """
    Trim conformers for the entire dataset.
    Args:
        dataset (nff.data.dataset): NFF dataset
        num_confs (int): desired number of conformers
        idx_dic (dict): Dicationary of the form
            {smiles: indices}, where `indices` are the indices
            of the conformers you want to keep. If not specified,
            then the top `num_confs` conformers with the highest
            statistical weight will be used.
        enum_func (callable, optional): a function with which to 
            enumerate the dataset. If not given, we use tqdm
            to track progress. 
    Returns:
        dataset (nff.data.dataset): updated NFF dataset
    """

    if enum_func is None:
        enum_func = tqdm_enum

    for i, batch in tqdm_enum(dataset):

        batch_dic = get_batch_dic(batch=batch,
                                  idx_dic=idx_dic,
                                  num_confs=num_confs)

        dataset = update_dset(batch=batch,
                              batch_dic=batch_dic,
                              dataset=dataset,
                              i=i)

    return dataset


def make_split_nbrs(nbr_list,
                    mol_size,
                    num_confs,
                    confs_per_split):
    """
    Split neighbor list of a species into chunks for each sub-batch.
    Args:
        nbr_list (torch.LongTensor): neighbor list for
        mol_size (int): number of atoms in the molecule
        num_confs (int): number of conformers in the species
        confs_per_split (list[int]): number of conformers in each
            sub-batch.
    Returns:
        all_grouped_nbrs (list[torch.LongTensor]): list of 
            neighbor lists for each sub-batch.
        nbr_masks (list(torch.BoolTensor))): masks that tell you which  
            indices of the combined neighbor list are being used for the  
            neighbor list of each sub-batch.
    """

    # first split by conformer
    new_nbrs = []
    masks = []

    for i in range(num_confs):
        # mask = (nbr_list[:, 0] <= (i + 1) * mol_size
        #         ) * (nbr_list[:, 1] <= (i + 1) * mol_size)

        mask = (nbr_list[:, 0] < (i + 1) * mol_size
                ) * (nbr_list[:, 0] >= i * mol_size)

        new_nbrs.append(nbr_list[mask])
        masks.append(mask)

    # regroup in sub-batches and subtract appropriately

    all_grouped_nbrs = []
    sub_batch_masks = []

    for i, num in enumerate(confs_per_split):

        # neighbor first
        prev_idx = sum(confs_per_split[:i])
        nbr_idx = list(range(prev_idx, prev_idx + num))

        grouped_nbrs = torch.cat([new_nbrs[i] for i in nbr_idx])
        grouped_nbrs -= mol_size * prev_idx

        all_grouped_nbrs.append(grouped_nbrs)

        # then add together all the masks
        mask = sum(masks[i] for i in range(prev_idx,
                                           prev_idx + num)).to(torch.bool)
        sub_batch_masks.append(mask)

    return all_grouped_nbrs, sub_batch_masks


def add_split_nbrs(batch,
                   mol_size,
                   num_confs,
                   confs_per_split,
                   sub_batches):
    """
    Add split-up neighbor lists to each sub-batch.
    Args:
        batch (dict): batched sample of species
        mol_size (int): number of atoms in the molecule
        num_confs (int): number of conformers in the species
        confs_per_split (list[int]): number of conformers in each
            sub-batch.
        sub_batches (list[dict]): list of sub_batches
    Returns:
        sub_batches (list[dict]): list of sub_batches updated with
            their neighbor lists.
        nbr_masks (list(torch.BoolTensor))): masks that tell you which  
            indices of the combined neighbor list are being used for the  
            neighbor list of each sub-batch.
    """

    # go through each key that needs to be reindex as a neighbor list
    # (i.e. the neighbor list and the bonded neighbor list)

    nbr_masks = None

    for key in REINDEX_KEYS:
        if key not in batch:
            continue
        nbr_list = batch[key]
        split_nbrs, masks = make_split_nbrs(nbr_list=nbr_list,
                                            mol_size=mol_size,
                                            num_confs=num_confs,
                                            confs_per_split=confs_per_split)
        if key == "nbr_list":
            nbr_masks = masks

        for i, sub_batch in enumerate(sub_batches):
            sub_batch[key] = split_nbrs[i]
            sub_batches[i] = sub_batch

    return sub_batches, nbr_masks


def get_confs_per_split(batch,
                        num_confs,
                        sub_batch_size):
    """
    Get the number of conformers per sub-batch.
    Args:
        batch (dict): batched sample of species
        num_confs (int): number of conformers in the species
        sub_batch_size (int): maximum number of conformers per
            sub-batch.
    Returns:
        confs_per_split (list[int]): number of conformers in each
            sub-batch.
    """

    val_len = len(batch["nxyz"])
    inherent_val_len = val_len // num_confs
    split_list = [sub_batch_size * inherent_val_len] * math.floor(
        num_confs / sub_batch_size)

    # if there's a remainder

    if sum(split_list) != val_len:
        split_list.append(val_len - sum(split_list))

    confs_per_split = [i // inherent_val_len for i in split_list]

    return confs_per_split


def fix_nbr_idx(batch,
                masks,
                sub_batches):
    """
    Fix anything that is defined with respect to positions
    of pairs in a neighbor list (e.g. `bond_idx`, `kj_idx`,
    and `ji_idx`).
    Args:
        batch (dict): batched sample of species
        masks (list(torch.BoolTensor))): masks that tell you which  
            indices of the combined neighbor list are being used for the  
            neighbor list of each sub-batch.
        sub_batches (list[dict]): sub batches of the batch
    Returns:
        sub_batches (list[dict]): corrected sub batches of the batch
    """

    old_nbr_list = batch['nbr_list']
    new_idx_list = []

    for mask in masks:
        num_new_nbrs = mask.nonzero().reshape(-1).shape[0]
        # make everything not in this batch equal to -1  so we
        # know what's actually not in this batch
        new_idx = -torch.ones_like(old_nbr_list)[:, 0]
        new_idx[mask] = (torch.arange(num_new_nbrs)
                         .to(mask.device))
        new_idx_list.append(new_idx)

    for new_idx, sub_batch in zip(new_idx_list, sub_batches):
        for key in NBR_IDX_KEYS:
            if key not in batch:
                continue
            new_val = new_idx[batch[key]]
            new_mask = new_val != -1
            new_val = new_val[new_mask].reshape(-1)

            sub_batch.update({key: new_val})

    return sub_batches


def split_batch(batch,
                sub_batch_size):
    """
    Split a batch into sub-batches.
    Args:
        batch (dict): batched sample of species
        sub_batch_size (int): maximum number of conformers per
            sub-batch.
    Returns:
        sub_batches (list[dict]): sub batches of the batch
    """

    mol_size = batch["mol_size"].item()
    num_confs = len(batch["nxyz"]) // mol_size
    sub_batch_dic = {}

    confs_per_split = get_confs_per_split(
        batch=batch,
        num_confs=num_confs,
        sub_batch_size=sub_batch_size)

    num_splits = len(confs_per_split)

    for key, val in batch.items():
        val_len = len(val)

        # save nbr lists for later and
        # get rid of `bond_idx` because it's wrong
        if key in [*NBR_IDX_KEYS, *REINDEX_KEYS]:
            continue
        elif np.mod(val_len, num_confs) != 0 or val_len == 1:
            if key == "num_atoms":
                sub_batch_dic[key] = [int(val * num / num_confs)
                                      for num in confs_per_split]
            else:
                sub_batch_dic[key] = [val] * num_splits
            continue

        # the per-conformer length of the value is `val_len`
        # divided by the number of conformers

        inherent_val_len = val_len // num_confs

        # use this to determine the number of items in each
        # section of the split list

        split_list = [inherent_val_len * num
                      for num in confs_per_split]

        # split the value accordingly
        split_val = torch.split(val, split_list)
        sub_batch_dic[key] = split_val

    sub_batches = [{key: sub_batch_dic[key][i] for key in
                    sub_batch_dic.keys()} for i in range(num_splits)]

    # fix neighbor list indexing
    sub_batches, masks = add_split_nbrs(batch=batch,
                                        mol_size=mol_size,
                                        num_confs=num_confs,
                                        confs_per_split=confs_per_split,
                                        sub_batches=sub_batches)

    # fix anything that relies on the position of a neighbor list pair
    sub_batches = fix_nbr_idx(batch=batch,
                              masks=masks,
                              sub_batches=sub_batches)

    return sub_batches
