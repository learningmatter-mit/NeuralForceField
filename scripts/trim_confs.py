"""
Script to copy a reference dataset into a new dataset
with fewer conformers per species.
"""


import torch
import os
import sys
import argparse
import json
import pdb
from tqdm import tqdm

from nff.data import Dataset


def fprint(msg):

    print(msg)
    sys.stdout.flush()


def assert_ordered(batch):

    # make sure the conformers are ordered by weight
    weights = batch["weights"].tolist()
    sort_weights = sorted(weights,
                          key=lambda x: -x)
    assert weights == sort_weights


def get_batch_dic(batch,
                  idx_dic,
                  num_confs):

    mol_size = batch["mol_size"]
    old_num_atoms = batch["num_atoms"]

    # if conformers in batch beforehand
    # are less than num_confs, use
    # the old number of conformers

    confs_in_batch = old_num_atoms // mol_size
    new_num_atoms = int(mol_size * min(
        confs_in_batch, num_confs))

    if idx_dic is None:

        assert_ordered(batch)
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

    confs_in_batch = batch_dic["confs_in_batch"]
    mol_size = batch_dic["mol_size"]
    conf_idx = batch_dic["conf_idx"]

    # the xyz indices of where each conformer starts
    xyz_conf_start_idx = [i * mol_size for i in range(confs_in_batch + 1)]
    # a list of the full set of indices for each conformer
    xyz_conf_all_idx = []

    for conf_num in conf_idx:

        start_idx = xyz_conf_start_idx[conf_num]
        end_idx = xyz_conf_start_idx[conf_num + 1]
        full_idx = torch.arange(start_idx, end_idx)

        xyz_conf_all_idx.append(full_idx)

    xyz_conf_all_idx = torch.cat(xyz_conf_all_idx)

    return xyz_conf_all_idx


def split_nbrs(nbrs,
               mol_size,
               confs_in_batch,
               conf_idx):

    split_idx = []
    cutoffs = [i * mol_size - 1 for i in range(1, confs_in_batch + 1)]

    for i in conf_idx:

        start = cutoffs[i] - mol_size
        end = cutoffs[i]
        mask = (nbrs[:, 0] <= end) * (nbrs[:, 1] <= end)
        mask *= (nbrs[:, 0] >= start) * (nbrs[:, 1] >= start)

        idx = mask.nonzero().reshape(-1)

        split_idx.append(idx)

    tens_idx = torch.cat(split_idx)

    return tens_idx


def to_nbr_idx(batch_dic, nbrs):

    mol_size = batch_dic["mol_size"]
    confs_in_batch = batch_dic["confs_in_batch"]
    conf_idx = batch_dic["conf_idx"]

    split_nbr_idx = split_nbrs(nbrs=nbrs,
                               mol_size=mol_size,
                               confs_in_batch=confs_in_batch,
                               conf_idx=conf_idx)

    return split_nbr_idx


def update_weights(batch, batch_dic):

    old_weights = batch["weights"]

    conf_idx = torch.LongTensor(batch_dic["conf_idx"])
    new_weights = old_weights[conf_idx]
    new_weights /= new_weights.sum()
    if torch.isnan(new_weights):
        new_weights = torch.ones_like(old_weights[conf_idx])
        new_weights /= new_weights.sum()
    return new_weights


def convert_nbrs(batch_dic, nbrs, nbr_idx):
    conf_idx = batch_dic["conf_idx"]
    mol_size = batch_dic["mol_size"]
    new_nbrs = []

    for i in range(len(conf_idx)):
        conf_id = conf_idx[i]
        delta = -conf_id * mol_size + i * mol_size
        new_nbrs.append(nbrs[nbr_idx] + delta)

    new_nbrs = torch.cat(new_nbrs)

    return new_nbrs


def update_dset(batch, batch_dic, dataset, i):

    bond_nbrs = batch["bonded_nbr_list"]
    nbr_list = batch["nbr_list"]
    bond_feats = batch["bond_features"]
    atom_feats = batch["atom_features"]
    nxyz = batch["nxyz"]

    conf_xyz_idx = to_xyz_idx(batch_dic)
    bond_nbr_idx = to_nbr_idx(batch_dic, bond_nbrs)
    all_nbr_idx = to_nbr_idx(batch_dic, nbr_list)

    dataset.props["num_atoms"][i] = batch_dic["new_num_atoms"]
    dataset.props["nxyz"][i] = nxyz[conf_xyz_idx]

    dataset.props["bonded_nbr_list"][i] = convert_nbrs(batch_dic,
                                                       bond_nbrs,
                                                       bond_nbr_idx)

    dataset.props["nbr_list"][i] = convert_nbrs(batch_dic,
                                                nbr_list,
                                                all_nbr_idx)

    dataset.props["bond_features"][i] = bond_feats[bond_nbr_idx]
    dataset.props["atom_features"][i] = atom_feats[conf_xyz_idx]
    dataset.props["weights"][i] = update_weights(batch,
                                                 batch_dic)

    return dataset


def tqdm_enum(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def trim_confs(dataset, num_confs, idx_dic):

    for i, batch in tqdm_enum(dataset):

        batch_dic = get_batch_dic(batch=batch,
                                  idx_dic=idx_dic,
                                  num_confs=num_confs)

        dataset = update_dset(batch=batch,
                              batch_dic=batch_dic,
                              dataset=dataset,
                              i=i)

    return dataset


def main(from_model_path,
         to_model_path,
         num_confs,
         conf_file,
         **kwargs):

    if conf_file is not None:
        with open(conf_file, "r") as f:
            idx_dic = json.load(f)
    else:
        idx_dic = None

    folders = sorted([i for i in os.listdir(from_model_path)
                      if i.isdigit()], key=lambda x: int(x))

    for folder in tqdm(folders):

        fprint(folder)
        for name in ["train.pth.tar", "test.pth.tar", "val.pth.tar"]:
            load_path = os.path.join(from_model_path, folder, name)
            if not os.path.isfile(load_path):
                continue
            dataset = Dataset.from_file(load_path)
            dataset = trim_confs(dataset=dataset,
                                 num_confs=num_confs,
                                 idx_dic=idx_dic)

            save_folder = os.path.join(to_model_path, folder)
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, name)
            dataset.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_model_path', type=str,
                        help="Path to model from which original data comes")
    parser.add_argument('--to_model_path', type=str,
                        help="Path to model to which new data is saved")
    parser.add_argument('--num_confs', type=int,
                        help="Number of conformers per species",
                        default=1)
    parser.add_argument('--conf_file', type=str,
                        help=("Path to json that says which conformer "
                              "to use for each species. This is optional. "
                              "If you don't specify the conformers, the "
                              "script will default to taking the `num_confs` "
                              "lowest conformers, ordered by statistical "
                              "weight."),
                        default=None)

    args = parser.parse_args()

    try:
        main(**args.__dict__)
    except Exception as e:
        fprint(e)
        pdb.post_mortem()
