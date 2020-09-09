import torch
import os
import sys
import argparse
import json
import math
import numpy as np
import copy
from tqdm import tqdm
import pdb

from nff.data.fp_cluster import ConfDataset, inner_opt
from nff.data import Dataset

# need to load the datasets that have both the nxyz and the fingerprints


def fprint(msg):

    print(msg)
    sys.stdout.flush()


def split_dset(dset, batch_size):

    ref_props = {key: val[:2] for key, val in dset.props.items()}
    ref_dset = Dataset(ref_props)

    length = len(dset)
    num_batches = math.ceil(length / batch_size)
    split_idx = [torch.LongTensor(idx) for idx in
                 np.array_split(list(range(length)), num_batches)]
    dsets = []
    for idx in split_idx:
        new_dset = copy.deepcopy(ref_dset)
        new_props = {}
        for key, val in dset.props.items():
            if type(val) is list:
                new_props[key] = [val[i] for i in idx]
            else:
                new_props[key] = val[idx]
        new_dset.props = new_props
        dsets.append(new_dset)

    return dsets


def update_conf_idx(dset, dic):

    new_dic = {smiles: idx for smiles, idx
               in zip(dset.props["smiles"], dset.props["conf_idx"])}

    for key, val in new_dic.items():

        # we must do this because we don't want to overwrite the conformers
        # we already know from `dset_1`, which will be different if we apply
        # the "most similar to dset_1" criterion

        if key in dic:
            continue

        if isinstance(val, torch.Tensor):
            val = val.tolist()

        dic[key] = [val]

    return dic


def add_fps(nff_dset,
            nff_dset_path,
            fp_dset_path):

    fprint("Adding fingerprints to dataset..")
    fp_dset = Dataset.from_file(fp_dset_path)

    smiles_dic = {smiles: i for i, smiles
                  in enumerate(fp_dset.props["smiles"])}
    fps = [fp_dset.props["fingerprint"][smiles_dic[smiles]]
           for smiles in nff_dset.props["smiles"]]

    nff_dset.props["fingerprint"] = fps

    fprint("Saving...")

    nff_dset.save(nff_dset_path)

    fprint("Done updating.")

    return nff_dset


def find_conf_idx(dset_1,
                  nff_dset,
                  batch_size,
                  idx_dic):

    nff_dsets = split_dset(nff_dset, batch_size)
    func_name = "cosine_similarity"

    for nff_dset_0 in tqdm(nff_dsets):

        conf_dset_0 = ConfDataset(nff_dset=nff_dset_0)
        conf_dset_0 = inner_opt(dset_0=conf_dset_0,
                                dset_1=dset_1,
                                func_name=func_name,
                                verbose=False,
                                debug=False)
        idx_dic = update_conf_idx(dset=conf_dset_0,
                                  dic=idx_dic)

    return idx_dic


def get_idx_dic(dset_1_path,
                nff_dset_paths,
                fp_dset_path,
                batch_size,
                update_fps):

    dset_1 = ConfDataset.from_file(dset_1_path)
    idx_dic = {}
    idx_dic = update_conf_idx(dset=dset_1,
                              dic=idx_dic)

    for nff_path in tqdm(nff_dset_paths):
        nff_dset = Dataset.from_file(nff_path)
        if "fingerprint" not in nff_dset.props or update_fps:
            nff_dset = add_fps(nff_dset=nff_dset,
                               nff_dset_path=nff_path,
                               fp_dset_path=fp_dset_path)

        idx_dic = find_conf_idx(dset_1=dset_1,
                                nff_dset=nff_dset,
                                batch_size=batch_size,
                                idx_dic=idx_dic)

    return idx_dic


def get_dset_paths(folder_path):
    nff_dset_paths = []
    names = ["train", "val", "test"]

    for name in names:
        dset_path = os.path.join(folder_path, name + ".pth.tar")
        nff_dset_paths.append(dset_path)

    return nff_dset_paths


def main(folder_path,
         dset_1_path,
         dic_save_path,
         fp_dset_path,
         batch_size,
         update_fps):

    # if os.path.isfile(dic_save_path):
    #     return

    nff_dset_paths = get_dset_paths(folder_path)
    idx_dic = get_idx_dic(dset_1_path=dset_1_path,
                          nff_dset_paths=nff_dset_paths,
                          fp_dset_path=fp_dset_path,
                          batch_size=batch_size,
                          update_fps=update_fps)

    with open(dic_save_path, "w") as f:
        json.dump(idx_dic, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str,
                        help="Folder with train/test/val files")
    parser.add_argument('--fp_dset_path', type=str,
                        help="Reference dataset with fingerprints")
    parser.add_argument('--dset_1_path', type=str,
                        help="Path ConfDataset optimized from MC")
    parser.add_argument('--dic_save_path', type=str,
                        help="Path to save dictionary with conformer idx")
    parser.add_argument('--batch_size', type=int,
                        help=("Batch size with which to "
                              "find optimized conformers"),
                        default=50)
    parser.add_argument('--update_fps', action="store_true",
                        help=("Update datasets with fingerprints "
                              "even if they already have them"),
                        default=False)

    args = parser.parse_args()
    try:
        main(**args.__dict__)
    except Exception as e:
        print(e)
        pdb.post_mortem()
