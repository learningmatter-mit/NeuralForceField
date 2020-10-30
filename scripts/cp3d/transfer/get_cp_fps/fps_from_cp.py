import os
import json
import numpy as np
import argparse

from nff.utils import bash_command, parse_args, fprint


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
               f"Save names: \n{string}\n")
    fprint(summary)


def main(cp_folder,
         feature_folder,
         model_folder_cp,
         model_names,
         device,
         smiles_folder,
         **kwargs):
    """
    Get fingerprints from a pre-trained chemprop model.
    Args:
      cp_folder (str): path to the chemprop folder on your computer
      feature_folder (str): general folder for features saving
      model_folder_cp (str): folder with the chemprop models
      model_names (list[str]): names of models from which we want the features
      device (Union[str, int]): device to evaluate the model on
      smiles_folder (str): folder with the csvs

    """

    script = os.path.join(cp_folder, "predict.py")
    save_paths = []

    for model_name in model_names:

        cp_model_path = os.path.join(model_folder_cp, model_name)

        # load the arguments from that model to get the features path
        args_path = f"{cp_model_path}/fold_0/args.json"
        if not os.path.isfile(args_path):
            args_path = args_path.replace("fold_0/", "")
        with open(args_path, "r") as f:
            args = json.load(f)
        features_path = args["separate_test_features_path"]

        # make a string for all of the checkpoint paths
        check_str = os.path.join(cp_model_path, "fold_0/model_0/model.pt")

        # make the chemprop command

        cmd = (f"python {script} "
               f" --checkpoint_paths {check_str} "
               f"--as_featurizer ")

        if device == "cpu":
            cmd += f" --no_cuda"
        else:
            cmd += f" --gpu {device} "

        if features_path is not None:
            cmd += f" --features_path {features_path[0]}"

        for split in ["train", "val", "test"]:

            feat_path = os.path.join(feature_folder,
                                     model_name,
                                     f"{split}.csv")
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

    for model_name in model_names:
        summarize(save_paths, os.path.join(feature_folder, model_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cp_folder', type=str,
                        help=("Path to the chemprop folder on your computer"))
    parser.add_argument('--feature_folder', type=str,
                        help="General folder in which you're saving features")
    parser.add_argument('--model_folder_cp', type=str,
                        help=("Folder with the various models"))
    parser.add_argument('--model_names', type=str, nargs="+",
                        help=("Names of models from which you want to "
                              "create fingerprints"))
    parser.add_argument('--device', type=str,
                        help=("Device to evaluate the model on"))
    parser.add_argument('--smiles_folder', type=str,
                        help=("Folder with the csvs"))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)

    main(**args.__dict__)
