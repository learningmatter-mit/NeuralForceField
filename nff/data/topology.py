"""
Script to generate topologies from a set of smiles strings and bond lists.
"""

import copy
import pdb
import torch
import numpy as np
import itertools

TOPOLOGIES = ["bonds", "angles", "dihedrals", "impropers", "pairs"]
ALL_TOPOLOGY_KEYS = [*TOPOLOGIES, *["num_{}".format(key) for key in TOPOLOGIES], "bonded_nbr_list"]
RE_INDEX_TOPOLOGY_KEYS = [*TOPOLOGIES, "bonded_nbr_list"]


def create_smiles_dic(props):

    """

    Generate a dictionary of smiles strings. The value of each string is itself a dictionary with information
    about the topology of that species, which will be updated later.
    Args:
        props (dict): property dictionary from nff.data.dataset.props
    Returns:
        full_dic (dict): dictionary described above.
    Example:
        props = {"num_atoms": [4], "smiles": "C", "nxyz": [[6.0, 0.0, 0.0, 0.0],
            [1.0, -0.5985, 0.2627, 0.875],
            [1.0, -0.5209, -0.7639, -0.5814],
            [1.0, 0.1508, 0.888, -0.6178],
            [1.0, 0.9686, -0.3868, 0.3242]]}
        create_smiles_dic(props)
        >> {'C': {'angles': None,
          'bonded_nbr_list': None,
          'bonds': None,
          'degree_vec': None,
          'dihedrals': None,
          'impropers': None,
          'neighbors': None,
          'num_angles': None,
          'num_atoms': 4,
          'num_bonds': None,
          'num_dihedrals': None,
          'num_impropers': None,
          'num_pairs': None,
          'pairs': None}}

    """

    # get the unique smiles strings in the properties
    unique_smiles = list(set(props["smiles"]))
    # initialize "sub_dic", a subdictionary for each smiles
    sub_dic = {"num_atoms": None, "bonded_nbr_list": None, "degree_vec": None, "neighbors": None,
        **{key: None for key in ALL_TOPOLOGY_KEYS}}
    # full_dic is what we will output
    full_dic = {key: copy.deepcopy(sub_dic) for key in unique_smiles}

    # update full_dic with the numbrer of atoms for each smiles
    for i, smiles in enumerate(props["smiles"]):
        if full_dic[smiles]["num_atoms"] is not None:
            continue
        full_dic[smiles]["num_atoms"] = props["num_atoms"][i]

    return full_dic

def set_preliminaries(bond_dic, smiles_dic):

    """
    Set the bonded neighbor list and degree vector for each smiles string. The degree vector
    is the number of bonds that each atom in a molecule has.

    Args:
        bond_dic (dict): a dictionary of the form {smiles: bonded neighbor list}
            for each smiles in the dataset.
        smiles_dic (dict): a smiles dictionary as defined in the function create_smiles_dic
    Returns:
        None
    """

    for smiles, sub_dic in smiles_dic.items():

        bonded_nbr_list = np.array(bond_dic[smiles])
        num_atoms = sub_dic["num_atoms"]
        # define A as an N x N zero matrix (N is the number of atoms)
        A = torch.zeros(num_atoms, num_atoms).to(torch.long)
        # every pair of indices with a bond then gets a 1
        A[bonded_nbr_list[:, 0], bonded_nbr_list[:, 1]] = 1
        # sum over one of the dimensions to get an array with bond number
        # for each atom
        d = A.sum(1)
        sub_dic["degree_vec"] = d 
        sub_dic["bonded_nbr_list"] = torch.tensor(bonded_nbr_list)

def unique_pairs(bonded_nbr_list):

    """
    Reduces the bonded neighbor list to only include unique pairs of bonds. For example,
    if atoms 3 and 5 are bonded, then `bonded_nbr_list` will have items [3, 5] and also
    [5, 3]. This function will reduce the pairs only to [3, 5] (i.e. only the pair in which
    the first index is lower).

    Args:
        bonded_nbr_list (list): list of arrays of bonded pairs for each molecule.
    Returns:
        sorted_pairs (list): same as bonded_nbr_list but without duplicate pairs.

    """

    unique_pairs = []
    for pair in bonded_nbr_list:
        # sort according to the first item in the pair
        sorted_pair = torch.sort(pair)[0].numpy().tolist()
        if sorted_pair not in unique_pairs:
            unique_pairs.append(sorted_pair)

    # now make sure that the sorting is still good (this may be unnecessary but I added
    # it just to make sure)
    idx = list(range(len(unique_pairs)))
    # first_arg = list of the the first node in each pair
    first_arg = [pair[0] for pair in unique_pairs]
    # sorted_idx = sort the indices of unique_pairs by the first node in each pair
    sorted_idx = [item[-1] for item in sorted(zip(first_arg, idx))]
    # re-arrange by sorted_idx
    sorted_pairs = torch.LongTensor(np.array(unique_pairs)[sorted_idx])

    return sorted_pairs

def set_bonds(smiles_dic):

    """
    Set the bonds between atoms.
    Args:
        None
    Returns:
        None
    """


    for smiles, sub_dic in smiles_dic.items():

        bonded_nbr_list = sub_dic["bonded_nbr_list"]
        degree_vec = sub_dic["degree_vec"]
        # get the unique set of bonded pairs
        bonds = unique_pairs(bonded_nbr_list)
        # neighbors is a list of bonded neighbor pairs for each atom.
        # Get it by splitting the bonded neighbor list by degree_vec
        neighbors = list(torch.split(bonded_nbr_list, degree_vec.tolist()))
        # second_arg_neighbors is just the second node in each set of bonded
        # neighbor pairs. Since the first node is already given implicitly by
        # the first index of `neighbors`, we don't need to use it anymore
        second_arg_neighbors = [neigbor[:, 1].tolist()
                                for neigbor in neighbors]
        # props["bonds"] is the full set of unique pairs of bonded atoms
        sub_dic["bonds"] = bonds
        # props["num_bonds"] is teh number of bonds
        sub_dic["num_bonds"] = torch.tensor(len(bonds))
        # props["neighbors"] is the `second_arg_neighbors` intrdoduced above.
        # Note that props["neighbors"] is for bonded neighbors, as opposed
        # to props["nbr_list"], which is just everything within a 5 A radius.
        sub_dic["neighbors"] = second_arg_neighbors

def set_angles(smiles_dic):

    """
    Set the angles among bonded atoms.
    Args:
        None
    Returns:
        None

    """

    for smiles, sub_dic in smiles_dic.items():

        neighbors = sub_dic["neighbors"]

        angles = [list(itertools.combinations(x, 2)) for x in neighbors]
        angles = [[[pair[0]]+[i]+[pair[1]] for pair in pairs]
                  for i, pairs in enumerate(angles)]
        angles = list(itertools.chain(*angles))
        angles = torch.LongTensor(angles)

        sub_dic["angles"] = angles
        sub_dic["num_angles"] = torch.tensor(len(angles))


def set_dihedrals(smiles_dic):

    """
    Set the dihedral angles among bonded atoms.
    Args:
        None
    Returns:
        None

    """

    for smiles, sub_dic in smiles_dic.items():

        neighbors = sub_dic["neighbors"]
        dihedrals = copy.deepcopy(neighbors)
        for i in range(len(neighbors)):
            for counter, j in enumerate(neighbors[i]):
                k = set(neighbors[i])-set([j])
                l = set(neighbors[j])-set([i])
                pairs = list(
                    filter(lambda pair: pair[0] < pair[1], itertools.product(k, l)))
                dihedrals[i][counter] = [[pair[0]]+[i]+[j]+[pair[1]]
                                         for pair in pairs]
        dihedrals = list(itertools.chain(
            *list(itertools.chain(*dihedrals))))
        dihedrals = torch.LongTensor(dihedrals)

        sub_dic["dihedrals"] = dihedrals
        sub_dic["num_dihedrals"] = torch.tensor(len(dihedrals))


def set_impropers(smiles_dic):

    """
    Set the improper angles among bonded atoms.
    Args:
        None
    Returns:
        None

    """
    for smiles, sub_dic in smiles_dic.items():
        neighbors = sub_dic["neighbors"]
        impropers = copy.deepcopy(neighbors)
        for i in range(len(impropers)):
            impropers[i] = [
                [i]+list(x) for x in itertools.combinations(neighbors[i], 3)]
        impropers = list(itertools.chain(*impropers))
        impropers = torch.LongTensor(impropers)

        sub_dic["impropers"] = impropers
        sub_dic["num_impropers"] = torch.tensor(len(impropers))

def set_pairs(smiles_dic, use_1_4_pairs):

    """
    Set the non-bonded pairs.
    Args:
        None
    Returns:
        None

    """

    for smiles, sub_dic in smiles_dic.items():

        bonds = sub_dic["bonds"]
        angles = sub_dic["angles"]
        dihedrals = sub_dic["dihedrals"]
        impropers = sub_dic["impropers"]
        num_atoms = sub_dic["num_atoms"]

        pairs = torch.eye(num_atoms, num_atoms)
        topologies = [bonds, angles, impropers]

        if use_1_4_pairs is False:
            topologies.append(dihedrals)
        for topology in topologies:
            for interaction_list in topology:
                for pair in itertools.combinations(interaction_list, 2):
                    pairs[pair[0], pair[1]] = 1
                    pairs[pair[1], pair[0]] = 1
        pairs = (pairs == 0).nonzero()
        pairs = pairs.sort(dim=1)[0].unique(dim=0).tolist()
        pairs = torch.LongTensor(pairs)

        sub_dic["pairs"] = pairs
        sub_dic["num_pairs"] = torch.tensor(len(pairs))

def smiles_dic_to_props(smiles_dic, props):

    """

    Add the information contained in smiles_dic to the existing props dictionary.
    Args:
        smiles_dic (dict): smiles dictionary as defined in the function create_smiles_dic
        props (dict): dictionary of properties from nff.data.dataset.props
    Returns:
        new_props (dict): updated dictionary of props with topology information about
            each species.

    """

    # the new keys to add are the topology keys, plus degree vec, which may be useful later
    new_keys = [*ALL_TOPOLOGY_KEYS, "degree_vec"]
    new_props = copy.deepcopy(props)
    new_props.update({key: [] for key in new_keys})

    for smiles in props["smiles"]:
        for key in new_keys:
            # update new_props with sub_dic, a dictionary with topology information about the
            # corresponding smiles
            sub_dic = smiles_dic[smiles]
            if key == "degree_vec":
                new_props[key].append(sub_dic[key].reshape(-1, 1))
            else:
                new_props[key].append(sub_dic[key])

    return new_props

def update_props_topologies(props, bond_dic, use_1_4_pairs=True):

    """
    Update properties with topologies.
    Args:
        props (dict): dictionary of properties from nff.data.dataset.props
        bond_dic (dict): a dictionary of the form {smiles: bonded neighbor list}
            for each smiles in the dataset.
        use_1_4_pairs (bool): use 1-4 pairs when considering non-bonded pairs
    Returns:
        new_props (dict): updated version of props with topology information

    Example:
        props = {"num_atoms": [5], "smiles": ["C"], "nxyz": [[[6.0, 0.0, 0.0, 0.0],
            [1.0, -0.5985, 0.2627, 0.875],
            [1.0, -0.5209, -0.7639, -0.5814],
            [1.0, 0.1508, 0.888, -0.6178],
            [1.0, 0.9686, -0.3868, 0.3242]]]}
        bond_dic = {"C": [[0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [2, 0], [3, 0], [4, 0]]}
        update_props_topologies(props, bond_dic)
        >> {'angles': [tensor([[1, 0, 2], [1, 0, 3], [1, 0, 4], [2, 0, 3], [2, 0, 4], [3, 0, 4]])],
        'bonds': [tensor([[0, 1], [0, 2], [0, 3], [0, 4]])], 'degree_vec': [tensor([4, 1, 1, 1, 1])],
        'dihedrals': [tensor([], dtype=torch.int64)], 'impropers': [tensor([[0, 1, 2, 3], [0, 1, 2, 4],
        [0, 1, 3, 4], [0, 2, 3, 4]])], 'num_angles': [tensor(6)], 'num_atoms': [5], 'num_bonds': [tensor(4)],
        'num_dihedrals': [tensor(0)], 'num_impropers': [tensor(4)], 'num_pairs': [None], 'nxyz':
        [[[6.0, 0.0, 0.0, 0.0], [1.0, -0.5985, 0.2627, 0.875], [1.0, -0.5209, -0.7639, -0.5814],
        [1.0, 0.1508, 0.888, -0.6178], [1.0, 0.9686, -0.3868, 0.3242]]], 'pairs': [None], 'smiles': ['C']}

    """

    smiles_dic = create_smiles_dic(props)
    set_preliminaries(bond_dic, smiles_dic)
    set_bonds(smiles_dic)
    set_angles(smiles_dic)
    set_dihedrals(smiles_dic)
    set_impropers(smiles_dic)
    set_pairs(smiles_dic, use_1_4_pairs)

    new_props = smiles_dic_to_props(smiles_dic=smiles_dic, props=props)

    return new_props


