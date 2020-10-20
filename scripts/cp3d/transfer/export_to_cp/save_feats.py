"""
Save fingerprints generated with a 3D model so that they can
be used as features for a ChemProp model. Also make a sub-split
of training and validation molecules for hyperparameter optimization.
"""

import os
import pickle
import numpy as np
import argparse
from nff.utils import parse_args, fprint


def get_smiles(smiles_folder, name):

    path = os.path.join(smiles_folder, name)
    with open(path, "r") as f:
        lines = f.readlines()
    smiles_list = [i.strip() for i in lines[1:]]
    return smiles_list


def save_smiles(smiles_folder, smiles_list, name):

    file_names = [f"{name}_smiles.csv", f"{name}_full.csv"]
    paths = [os.path.join(smiles_folder, name) for name in
             file_names]
    for path in paths:
        with open(path, "r") as f:
            lines = f.readlines()
        keep_lines = [lines[0]]
        for line in lines[1:]:
            smiles = line.split(",")[0].strip()
            if smiles in smiles_list:
                keep_lines.append(line)
        text = "".join(keep_lines)
        with open(path, "w") as f:
            f.write(text)


def get_name(feat_folder):
    splits = feat_folder.split("/")
    if splits[-1] == "":
        name = splits[-2]
    else:
        name = splits[-1]
    return name


def summarize(save_paths, feat_folder):

    base_dir = "/".join(save_paths[0].split("/")[:-1])
    save_names = [get_name(path) for path in save_paths]
    num_files = len(save_paths)
    string = "\n".join(save_names)
    summary = (f"Saved {num_files} files with features \n"
               f"Used model in {feat_folder} \n\n"
               f"Save folder: \n{base_dir}\n\n"
               f"Save names: \n{string}")
    fprint(summary)


def save_hyperopt(feat_folder,
                  metric,
                  smiles_folder,
                  cp_save_folder,
                  dset_size):
    """
    Aggregate the train and validation SMILES for hyperparameter optimization.
    Split them into train/val/test for use in hyperopt.
    """

    names = ["train", "val"]
    all_feats = []
    all_smiles = []

    for name in names:
        smiles_list = get_smiles(smiles_folder, f"{name}_smiles.csv")
        np_save_path = os.path.join(cp_save_folder,
                                    f"{name}_{metric}.npz")
        feats = np.load(np_save_path)
        all_feats.append(feats)
        all_smiles += smiles_list

    all_feats = np.concatenate(all_feats)

    if dset_size is not None:
        all_smiles = all_smiles[:dset_size]
        all_feats = all_feats[:dset_size]

    # save the entire train + val dataset for hyperopt, and let chemprop
    # do the subsequent splitting during hyperopt

    hyp_np_path = os.path.join(cp_save_folder,
                               f"hyperopt_{metric}.npz")
    np.savez_compressed(hyp_np_path, features=all_feats)
    save_smiles(smiles_folder, all_smiles, name="hyperopt")


def main(feat_folder,
         metrics,
         smiles_folder,
         cp_save_folder,
         hyper_dset_size,
         **kwargs):

    save_paths = []
    for metric in metrics:
        names = ["train", "val", "test"]

        for name in names:
            file_name = f"pred_{metric}_{name}.pickle"
            file_path = os.path.join(feat_folder, file_name)
            with open(file_path, "rb") as f:
                dic = pickle.load(f)

            smiles_list = get_smiles(smiles_folder, f"{name}_smiles.csv")
            smiles_list = [smiles for smiles in smiles_list if smiles in dic]

            # re-save the SMILES strings to account for any that we may
            # not be using
            save_smiles(smiles_folder, smiles_list, name)

            # arrange the features in that order
            ordered_feats = np.stack([dic[smiles]["fp"]
                                      for smiles in smiles_list])

            if not os.path.isdir(cp_save_folder):
                os.makedirs(cp_save_folder)

            np_save_path = os.path.join(cp_save_folder,
                                        f"{name}_{metric}.npz")

            np.savez_compressed(np_save_path, features=ordered_feats)
            save_paths.append(np_save_path)

        save_hyperopt(feat_folder=feat_folder,
                      metric=metric,
                      smiles_folder=smiles_folder,
                      cp_save_folder=cp_save_folder,
                      dset_size=hyper_dset_size)

    summarize(save_paths, feat_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--feat_folder', type=str,
                        help=("Path to model folder where fingerprint "
                              "pickles are saved"))
    parser.add_argument('--cp_save_folder', type=str,
                        help=("Path in which you'll save the ChemProp-"
                              "readable fingerprints"))
    parser.add_argument('--smiles_folder', type=str,
                        help=("Path to model folder where SMILES strings "
                              "are saved. These should be called "
                              "train_smiles.csv, val_smiles.csv, and "
                              "test_smiles.csv, as they are if generated "
                              "by a ChemProp split."))
    parser.add_argument('--metrics', type=str, nargs='+',
                        help=("Generate features from 3D models "
                              "whose best model is chosen according to "
                              "these metrics."))
    parser.add_argument('--hyper_dset_size', type=str, default=None,
                        help=("Maximum size of the subset of the data  "
                              "used for hyperparameter optimization. "
                              "If not specified, the  entire training "
                              "and validation set will be used."))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)

    main(**args.__dict__)
