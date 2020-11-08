"""
Tools for manipulating conformer numbers in a dataset.
"""

import torch
from nff.utils.misc import tqdm_enum


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
    if torch.isnan(new_weights):
        new_weights = torch.ones_like(old_weights[conf_idx])
        new_weights /= new_weights.sum()
    return new_weights


def convert_nbrs(batch_dic, nbrs, nbr_idx):
    """
    Convert neighbor list to only include neighbors from the conformers
    we're looking at.
    Args:
        batch_dic (dict): Dictionary with extra conformer 
            information about the batch
        nbrs (torch.LongTensor): neighbor list
        nbr_idx (torch.LongTensor): nbr indices of conformers we're
            keeping. 
    Returns:
        new_nbrs (torch.LongTensor): updated neighbor list

    """

    conf_idx = batch_dic["conf_idx"]
    mol_size = batch_dic["mol_size"]
    new_nbrs = []

    for i in range(len(conf_idx)):
        conf_id = conf_idx[i]
        delta = -conf_id * mol_size + i * mol_size
        new_nbrs.append(nbrs[nbr_idx] + delta)

    new_nbrs = torch.cat(new_nbrs)

    return new_nbrs


def update_bond_idx(batch, new_nbrs):
    """
    Update `bond_idx` (which are the mappings of bonded atom pairs
    to their locations in the neighbor list) based on the trimming of
    the neighbor list.
    Args:
        batch (dict): Batch dictionary
        new_nbrs (torch.LongTensor): Trimmed neighbor list
    Returns:
        new_bond_idx (torch.LongTensor): updated bond idx.

    """
    bond_idx = batch["bond_idx"]
    nbr_shape = new_nbrs.shape[0]
    mask = bond_idx < nbr_shape
    new_bond_idx = bond_idx[mask]

    return new_bond_idx


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
    dataset.props["num_atoms"][i] = batch_dic["new_num_atoms"]
    # get the right nxyz
    dataset.props["nxyz"][i] = nxyz[conf_xyz_idx]

    # convert the neighbor lists
    dataset.props["bonded_nbr_list"][i] = convert_nbrs(batch_dic,
                                                       bond_nbrs,
                                                       bond_nbr_idx)
    new_nbrs = convert_nbrs(batch_dic,
                            nbr_list,
                            all_nbr_idx)
    dataset.props["nbr_list"][i] = new_nbrs

    # get the atom and bond features at the right indices
    dataset.props["bond_features"][i] = bond_feats[bond_nbr_idx]
    dataset.props["atom_features"][i] = atom_feats[conf_xyz_idx]

    # renormalize weights
    dataset.props["weights"][i] = update_weights(batch,
                                                 batch_dic)

    # get rid of any entries in `bond_idx` that don't exist anymore
    if "bond_idx" in dataset.props:
        dataset.props["bond_idx"][i] = update_bond_idx(batch,
                                                       new_nbrs)

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
