import torch
import copy
import numpy as np
import yaml

from nff.data import Dataset

SMILES_PROPS = ['sars_cov_one_cl_protease_active', 'poplowestpct', 'totalconfs',
                'uniqueconfs', 'ensembleenergy', 'charge', 'ensembleentropy', 'ensemblefreeenergy', 'lowestenergy',
                'smiles', 'sars_cov_one_pl_protease_active', 'pseudomonas_active', 'ecoli_inhibitor']

CONF_PROPS = ['degeneracy', 'energy', 'weights']


def split_nxyz(dic):

    mol_size = dic['mol_size'].item()
    nxyz = dic['nxyz']
    num_mols = len(nxyz) // mol_size
    split = torch.split(nxyz, [mol_size] * num_mols)

    return split, num_mols


def split_nbrs(dic, dic_list, num_mols):

    num_bonds = dic['num_bonds']
    if type(num_bonds) is not list:
        num_bonds = num_bonds.numpy().astype('int').tolist()
    mol_size = dic['mol_size'].item()

    working_nbrs = copy.deepcopy(dic['nbr_list'])
    working_bond_nbrs = copy.deepcopy(dic['bonded_nbr_list'])
    split_nbrs = []
    split_bond_nbrs = []

    for conf in range(num_mols):
        nbr_mask = (working_nbrs[:, 0] < mol_size) * (working_nbrs[:, 1] < mol_size
                                                      ) * (working_nbrs[:, 0] >= 0) * (working_nbrs[:, 1] >= 0)
        split_nbrs.append(working_nbrs[nbr_mask].tolist())
        working_nbrs -= mol_size

        nbr_mask = (working_bond_nbrs[:, 0] < mol_size) * (working_bond_nbrs[:, 1] < mol_size
                                                           ) * (working_bond_nbrs[:, 0] >= 0) * (working_bond_nbrs[:, 1] >= 0)
        split_bond_nbrs.append(working_bond_nbrs[nbr_mask].tolist())
        working_bond_nbrs -= mol_size

    split_bond_features = torch.split(dic['bond_features'], num_bonds)
    split_atom_features = torch.split(
        dic['atom_features'], [dic['mol_size'].item()] * num_mols)

    for i, sub_dic in enumerate(dic_list):
        sub_dic.update({"nbr_list": split_nbrs[i],
                        "bonded_nbr_list": split_bond_nbrs[i],
                        "bond_features": split_bond_features[i].tolist(),
                        "atom_features": split_atom_features[i].tolist()})
        dic_list[i] = sub_dic

    return dic_list


def split_confs(dic, use_features):

    split, num_mols = split_nxyz(dic)
    weights = dic['weights']
    if torch.tensor(weights.shape).prod().item() == 1:
        weights = [weights]

    dic_list = [{"xyz": split[i].tolist(), **{key: dic[key][i].item()
                                              for key in CONF_PROPS}} for i in range(len(weights))]

    for i, sub_dic in enumerate(dic_list):
        sub_dic['weight'] = sub_dic['weights']
        sub_dic.pop('weights')
        dic_list[i] = sub_dic

    if 'bonded_nbr_list' in dic and use_features:
        dic_list = split_nbrs(dic=dic, dic_list=dic_list, num_mols=num_mols)

    # import pdb
    # pdb.set_trace()

    # import pdb
    # pdb.set_trace()
    dic.update({"conformers": dic_list})

    return dic


def make_smiles_dic(dic, use_features):

    # import pdb
    # pdb.set_trace()

    dic = split_confs(dic, use_features)

    # import pdb
    # pdb.set_trace()

    for key in dic.keys():
        val = dic[key]
        if hasattr(val, "item"):
            if torch.prod(torch.tensor(val.shape)) == 1:
                val = val.item()
                if np.isnan(val):
                    val = None
            else:
                val = val.tolist()
        dic[key] = val

    keys = list(dic.keys())
    for key in keys:
        if key not in SMILES_PROPS and key != 'conformers':
            dic.pop(key)

    for key in SMILES_PROPS:
        if key not in dic:
            dic[key] = None

    return dic


def convert_data(dataset, use_features):
    new_data = []
    for dic in dataset:
        new_dic = make_smiles_dic(dic, use_features)
        new_data.append(new_dic)

    # convert to dictionary
    final_data = {dic['smiles']:
                  {key: val for key, val in dic.items() if key != 'smiles'} for dic in new_data}

    return final_data

def save_data(dataset, use_features, save_path):
    new_data = convert_data(dataset, use_features=use_features)
    with open(save_path, "w") as f:
        yaml.dump(new_data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    path = '/home/saxelrod/engaging_nfs/data_from_fock/combined_fingerprint_datasets/combined_dset_246.pth.tar'
    save_path = "/home/saxelrod/json_test.yml"
    dataset = Dataset.from_file(path)
    use_features = False

    save_data(dataset, use_features, save_path)


