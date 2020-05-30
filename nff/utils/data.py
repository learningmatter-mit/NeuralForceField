import torch
import numpy as np
import copy
import pickle
from nff.data import Dataset, concatenate_dict


def prepare_conf_list(sub_dic):

    conf_list = sub_dic['conformers']
    new_conf_list = []

    for i, dic in enumerate(conf_list):
        new_dic = copy.deepcopy(dic)
        for key, val in dic.items():
            if isinstance(val, list):
                val = torch.Tensor(val)
                new_dic[key] = val
        new_conf_list.append(new_dic)
    return new_conf_list


def concat_conformers(sub_dic, nbrlist_cutoff=5.0):

    conf_list = prepare_conf_list(sub_dic)
    props = concatenate_dict(*conf_list)

    if "xyz" in props:
        props["nxyz"] = props["xyz"]
        props.pop("xyz")

    dataset = Dataset(props.copy(), units='kcal/mol')
    mol_size = len(dataset.props["nxyz"][0])
    dataset.generate_neighbor_list(cutoff=nbrlist_cutoff)

    # now combine the neighbor lists so that this set
    # of nxyz's can be treated like one big molecule

    nbrs = dataset.props['nbr_list']
    # number of atoms in the molecule
    new_nbrs = []

    # shift by i * mol_size for each conformer
    for i in range(len(nbrs)):
        new_nbrs.append(nbrs[i] + i * mol_size)

    # add to list of conglomerated neighbor lists
    new_nbrs = torch.cat(new_nbrs)
    dataset.props.pop('nbr_list')

    # concatenate the nxyz's
    nxyz = np.concatenate([np.array(item) for item in props["nxyz"]]
                          ).reshape(-1, 4).tolist()

    new_dic = {}
    for key, val in dataset.props.items():
        if key == "nxyz":
            continue
        if isinstance(val, torch.Tensor):
            val = val.to(torch.float)
        try:
            new_dic[key] = torch.Tensor(val).reshape(-1, 1)
        except ValueError:
            new_dic[key] = val

    new_dic.update({"mol_size": mol_size,
                    "nxyz": nxyz,
                    "num_atoms": [len(nxyz)],
                    "nbr_list": new_nbrs})

    return new_dic


def from_db_pickle(path, nbrlist_cutoff):

    with open(path, "rb") as f:
        dic = pickle.load(f)

    props_list = []

    for smiles, sub_dic in dic.items():
        concat_dic = concat_conformers(sub_dic, nbrlist_cutoff)
        props_list.append({"smiles": smiles, **concat_dic})

    props = concatenate_dict(*props_list)
    dataset = Dataset(props=props, units='kcal/mol')

    return dataset


def get_bond_list(mol):

    bond_list = []
    bonds = mol.GetBonds()

    for bond in bonds:

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        lower = min((start, end))
        upper = max((start, end))
        bond_name = bond.GetBondType().name.lower()

        bond_list.append({"indices": [lower, upper],
                          "type": bond_name})

    return bond_list

def split_nxyz(dic):
    
    mol_size = dic['mol_size'].item()
    nxyz = dic['nxyz']
    num_mols = len(nxyz) // mol_size
    split = torch.split(nxyz, [mol_size] * num_mols)
    
    return split, num_mols

def split_confs(dic):
    
    split, num_mols = split_nxyz(dic)   
    mol_size = dic['mol_size'].item()
    
    working_nbrs = copy.deepcopy(dic['nbr_list'])
    split_nbrs = []
    
    for conf in range(num_mols):
        nbr_mask = (working_nbrs[:, 0] < mol_size) * (working_nbrs[:, 1] < mol_size
                    ) * (working_nbrs[:, 0] >= 0) * (working_nbrs[:, 1] >= 0)
        split_nbrs.append(working_nbrs[nbr_mask].tolist())
        working_nbrs -= mol_size
      
    return split_nbrs

