"""
Save fingerprints generated with a 3D model so that they can
be used as features for a ChemProp model.
"""

import os
import pickle
import numpy as np
import argparse
from nff.utils import parse_args, fprint

METRICS = ["loss", "roc_auc", "prc_auc"]


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


def main(feat_folder,
         metrics,
         smiles_folder,
         cp_save_folder,
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

            # re-save the SMILES strings to make sure they're in the same
            # order as the features
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
                              "these metrics."),
                        default=METRICS)
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)

    main(**args.__dict__)
