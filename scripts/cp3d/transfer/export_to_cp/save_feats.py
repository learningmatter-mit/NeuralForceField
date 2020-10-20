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
    """
    Get SMILES strings from csv.
    Args:
      smiles_folder (str): folder with the csvs
      name (str): csv file name
    Returns:
      smiles_list (list[str]): SMILES strings 
    """

    path = os.path.join(smiles_folder, name)
    with open(path, "r") as f:
        lines = f.readlines()
    smiles_list = [i.strip() for i in lines[1:]]
    return smiles_list


def save_smiles(smiles_folder, smiles_list, name):
    """
    Re-save the SMILES strings, ignoring those that aren't in
    `smiles_list`.

    Args:
      smiles_folder (str): folder with the csvs
      smiles_list (list[str]): SMILES strings that we will use
        in training -- excludes those that, for example, do not
        have 3D structures.
      name (str): csv file name

    Returns:
      None

    """

    # both the file with only the SMILES string, and the file
    # that has the SMILES string with its properties (e.g. bind /
    # no bind):

    file_names = [f"{name}_smiles.csv", f"{name}_full.csv"]
    paths = [os.path.join(smiles_folder, name) for name in
             file_names]
    for path in paths:
        with open(path, "r") as f:
            lines = f.readlines()
        keep_lines = [lines[0]]
        for line in lines[1:]:
            smiles = line.split(",")[0].strip()

            # keep the line only if the SMILES string is in
            # `smiles_list`

            if smiles in smiles_list:
                keep_lines.append(line)
        text = "".join(keep_lines)
        with open(path, "w") as f:
            f.write(text)


def get_name(path):
    """
    Get a file name from its full path.
    Args:
      path (str): full path
    Returns:
      name (str): just the file name
    """

    splits = path.split("/")
    if splits[-1] == "":
        name = splits[-2]
    else:
        name = splits[-1]
    return name


def summarize(save_paths, feat_folder):
    """
    Summarize where the files were saved and what their contents are.
    Args:
      save_paths (list[str]): list of the paths to all the saved features files
      feat_folder (str): path to the folder that contains all the feature files.
    Returns:
      None
    """

    base_dir = "/".join(save_paths[0].split("/")[:-1])
    save_names = [get_name(path) for path in save_paths]
    num_files = len(save_paths)
    string = "\n".join(save_names)
    summary = (f"Saved {num_files} files with features \n"
               f"Used model in {feat_folder} \n\n"
               f"Save folder: \n{base_dir}\n\n"
               f"Save names: \n{string}")
    fprint(summary)


def make_hyperopt_csvs(smiles_folder, all_smiles):
    """
    Make csv files with SMILES strings for hyperparameter optimization.
    Args:
      smiles_folder (str): folder with the csvs
      all_smiles (list[str]): combined train and val SMILES for hyperparameter
        optimization that are actually used
    Returns:
      None
    """

    # full csv with properties, and just smiles csv
    suffixes = ["smiles", "full"]
    # dictionary with the combined lines read from train and val csvs
    # for each of the suffixes
    combined_lines = {suffix: [] for suffix in suffixes}

    for i, name in enumerate(["train", "val"]):
        for suffix in suffixes:
            file_path = os.path.join(smiles_folder, f"{name}_{suffix}.csv")
            with open(file_path, "r") as f:
                lines = f.readlines()

            # only include the header in the first file
            if i != 0:
                lines = lines[1:]
            combined_lines[suffix] += lines

    # write to new hyperopt csvs
    for suffix, lines in combined_lines.items():
        text = "".join(lines)
        new_path = os.path.join(smiles_folder, f"hyperopt_{suffix}.csv")
        with open(new_path, "w") as f:
            f.write(text)

    # re-save to account for the fact that not all smiles are used
    save_smiles(smiles_folder, all_smiles, name="hyperopt")


def save_hyperopt(feat_folder,
                  metric,
                  smiles_folder,
                  cp_save_folder,
                  dset_size):
    """
    Aggregate and save the train and validation SMILES for hyperparameter optimization.
    Args:
      feat_folder (str): path to the folder that contains all the feature files.
      metric (str): metric with which you're evaluating the model performance
      smiles_folder (str): folder with the csvs
      cp_save_folder (str): folder in which you're saving features for chemprop use
      dset_size (int, optional): maximum size of the entire dataset to use in hyperparameter 
        optimization. 
    Returns:
      None
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

    # save the entire train + val dataset features
    hyp_np_path = os.path.join(cp_save_folder,
                               f"hyperopt_{metric}.npz")
    np.savez_compressed(hyp_np_path, features=all_feats)

    # save csvs for the train + val dataset
    make_hyperopt_csvs(smiles_folder=smiles_folder,
                       all_smiles=all_smiles)


def main(feat_folder,
         metrics,
         smiles_folder,
         cp_save_folder,
         hyper_dset_size,
         **kwargs):
    """
    Save features from CP3D model for hyperopt and training with ChemProp.
    Args:
      feat_folder (str): path to the folder that contains all the feature files.
      metrics (list[str]): metrics with which you'll evaluate the model performance
      smiles_folder (str): folder with the csvs
      cp_save_folder (str): folder in which you're saving features for chemprop use
      hyper_dset_size (int, optional): maximum size of the entire dataset to use in hyperparameter 
        optimization. 
    Returns:
      None
    """

    save_paths = []
    for metric in metrics:
        names = ["train", "val", "test"]

        for name in names:
            # load the features from a pickle file according to the metric
            # being used
            file_name = f"pred_{metric}_{name}.pickle"
            file_path = os.path.join(feat_folder, file_name)
            with open(file_path, "rb") as f:
                dic = pickle.load(f)

            # get the smiles strings that are both in the csv and in the
            # feature dictionary
            smiles_list = get_smiles(smiles_folder, f"{name}_smiles.csv")
            smiles_list = [smiles for smiles in smiles_list if smiles in dic]

            # re-save the SMILES strings to account for any that we may
            # not be using
            save_smiles(smiles_folder, smiles_list, name)

            # arrange the features in the order of SMILES strings in the csv
            ordered_feats = np.stack([dic[smiles]["fp"]
                                      for smiles in smiles_list])

            if not os.path.isdir(cp_save_folder):
                os.makedirs(cp_save_folder)

            # save the features
            np_save_path = os.path.join(cp_save_folder,
                                        f"{name}_{metric}.npz")

            np.savez_compressed(np_save_path, features=ordered_feats)
            save_paths.append(np_save_path)

        # save the subset of train and val for hyperparameter optimization
        save_hyperopt(feat_folder=feat_folder,
                      metric=metric,
                      smiles_folder=smiles_folder,
                      cp_save_folder=cp_save_folder,
                      dset_size=hyper_dset_size)

    # tell the user what we did
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
