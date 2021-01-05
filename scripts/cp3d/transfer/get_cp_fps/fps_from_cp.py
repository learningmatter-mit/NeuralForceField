import os
import json
import numpy as np
import argparse

from nff.utils import (bash_command, parse_args, fprint,
                       CHEMPROP_METRICS)
from nff.io.cprop import save_hyperopt


def to_npz(csv):
    """
    Convert csv features to npz features.
    Args:
      csv (str): path to csv file
    Returns:
      np_path (str): path to saved file
    """
    with open(csv, "r") as f:
        lines = f.readlines()[1:]

    feats = []
    for line in lines:
        these_feats = (np.array(line.strip().split(",")[1:]
                                ).astype("float"))
        feats.append(these_feats)

    np_path = csv.replace(".csv", ".npz")
    np.savez_compressed(np_path, features=np.array(feats))

    return np_path


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
      feat_folder (str): path to the folder that contains all the feature
      files.
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
               f"Save names: \n{string}\n")
    fprint(summary)


def main(cp_folder,
         feature_folder,
         model_folder_paths,
         device,
         smiles_folder,
         metrics,
         hyper_dset_size,
         **kwargs):
    """
    Get fingerprints from a pre-trained chemprop model.
    Args:
      cp_folder (str): path to the chemprop folder on your computer
      feature_folder (str): folder in which you're sainvg the features
      model_folder_paths (str): folders with the different models from which
        you're making fingerprints.
      device (Union[str, int]): device to evaluate the model on
      smiles_folder (str): folder with the csvs
      metrics (list[str]): names of the metrics corresponding to each
        ChemProp model name.
      hyper_dset_size (int): maximum size of the entire dataset to use in
        hyperparameter optimization.

    """

    script = os.path.join(cp_folder, "predict.py")
    save_paths = []
    hyper_paths = []

    for model_path, metric in zip(model_folder_paths, metrics):

        # load the arguments from that model to get the features path
        args_path = f"{model_path}/fold_0/args.json"
        if not os.path.isfile(args_path):
            args_path = args_path.replace("fold_0/", "")
        with open(args_path, "r") as f:
            args = json.load(f)
        features_path = args["separate_test_features_path"]

        # make a string for all of the checkpoint paths
        check_str = os.path.join(model_path, "fold_0/model_0/model.pt")

        # make the chemprop command

        cmd = ("source activate chemprop && "
               f"python {script} "
               f" --checkpoint_paths {check_str} "
               f"--as_featurizer ")

        if device == "cpu":
            cmd += f" --no_cuda"
        else:
            cmd += f" --gpu {device} "

        if features_path is not None:
            feat_str = " ".join(features_path)
            cmd += f" --features_path {feat_str}"

        for split in ["train", "val", "test"]:

            feat_path = os.path.join(feature_folder,
                                     f"{split}_{metric}.csv")
            data_path = os.path.join(smiles_folder, f"{split}_full.csv")

            if not os.path.isdir(feature_folder):
                os.makedirs(feature_folder)

            cmd += (f" --test_path {data_path} "
                    f" --preds_path {feat_path} ")

            p = bash_command(cmd)
            p.wait()

            # convert it to npz
            np_save_path = to_npz(feat_path)
            save_paths.append(np_save_path)

        # make hyperparameter optimization splits
        hyp_save_path = save_hyperopt(feat_folder=feature_folder,
                                      metric=metric,
                                      smiles_folder=smiles_folder,
                                      cp_save_folder=feature_folder,
                                      dset_size=hyper_dset_size)
        hyper_paths.append(hyp_save_path)

    summarize(save_paths + hyper_paths, feature_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cp_folder', type=str,
                        help=("Path to the chemprop folder on your computer"))
    parser.add_argument('--feature_folder', type=str,
                        help="Folder in which you're saving the features")
    parser.add_argument('--model_folder_paths', type=str, nargs="+",
                        help=("Folders with the different models from which "
                              "you're making fingerprints. Each corresponds "
                              "to a model trained on the same data with a "
                              "different metric."))
    parser.add_argument('--metrics', type=str, nargs="+",
                        choices=CHEMPROP_METRICS,
                        help=("Metrics of each ChemProp model path."))
    parser.add_argument('--device', type=str,
                        help=("Device to evaluate the model on"))
    parser.add_argument('--smiles_folder', type=str,
                        help=("Folder with the csvs"))
    parser.add_argument('--hyper_dset_size', type=int,
                        help=("Maximum size of the entire dataset to use in "
                              "hyperparameter optimization. "))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)

    main(**args.__dict__)
