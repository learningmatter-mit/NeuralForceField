"""
Save fingerprints generated with a 3D model so that they can
be used as features for a ChemProp model. Also make a sub-split
of training and validation molecules for hyperparameter optimization.
"""

import os
import pickle
import numpy as np
import argparse

from nff.io.cprop import get_smiles, save_smiles, save_hyperopt
from nff.utils import parse_args, fprint


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


def main(feat_folder,
         metrics,
         smiles_folder,
         cp_save_folder,
         hyper_dset_size,
         **kwargs):
    """
    Save features from CP3D model for hyperopt and training with ChemProp.
    Args:
      feat_folder (str): Path to model folder where fingerprint pickles are saved
      metrics (list[str]): metrics with which you'll evaluate the model performance
      smiles_folder (str): folder with the csvs
      cp_save_folder (str): folder in which you're saving features for chemprop use
      hyper_dset_size (int): maximum size of the entire dataset to use in hyperparameter 
        optimization. 
    Returns:
      None
    """

    save_paths = []
    hyp_save_paths = []
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
        hyp_save_path = save_hyperopt(feat_folder=feat_folder,
                                      metric=metric,
                                      smiles_folder=smiles_folder,
                                      cp_save_folder=cp_save_folder,
                                      dset_size=hyper_dset_size)
        hyp_save_paths.append(hyp_save_path)

    # tell the user what we did
    summarize(save_paths + hyp_save_paths, feat_folder)


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
                              "these metrics. You can choose any of the "
                              "metrics that you tracked during training. "
                              "If you trained a classifier, you can also "
                              "request `binary_cross_entropy`, and you will "
                              "get the model with the best loss. If "
                              "you trained a regressor, you can also "
                              "request `mse`, and you will get the model "
                              "with the best loss."))
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
