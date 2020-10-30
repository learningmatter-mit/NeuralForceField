"""
Wrapper to get the ChemProp predictions on the test set for each model in
the folder `model_folder_cp`.
"""

import os
import json
import argparse

from nff.utils import bash_command, parse_args


def predict(cp_folder,
            test_path,
            cp_model_path,
            device):
    """
    Get and save the prediction results from a ChemProp model.
    Args:
      cp_folder (str): path to the chemprop folder on your computer
      test_path (str): path to the file with the test SMILES and their properties
      cp_model_path (str): path to the folder with the model of interest
      device (Union[str, int]): device to evaluate the model on
    Returns:
      None
    """

    script = os.path.join(cp_folder, "predict.py")
    preds_path = os.path.join(cp_model_path, f"test_pred.csv")

    # get the paths of all the models saved with different initial random seeds
    check_names = [i for i in os.listdir(cp_model_path)
                   if i.startswith("fold_") and i.split("_")[-1].isdigit()]
    check_paths = [os.path.join(cp_model_path, name, "model_0/model.pt")
                   for name in check_names]

    # if there are none, then this is not actually a model path
    if len(check_paths) == 0:
        return

    # load the arguments from that model to get the features path
    args_path = f"{cp_model_path}/fold_0/args.json"
    if not os.path.isfile(args_path):
        args_path = args_path.replace("fold_0/", "")
    with open(args_path, "r") as f:
        args = json.load(f)
    features_path = args["separate_test_features_path"]

    # make a string for all of the checkpoint paths
    check_str = " ".join(check_paths)

    # make the chemprop command

    cmd = (f"python {script} "
           f" --test_path {test_path} --preds_path {preds_path} "
           f" --checkpoint_paths {check_str} ")

    if device == "cpu":
        cmd += f" --no_cuda"
    else:
        cmd += f" --gpu {device} "

    if features_path is not None:
        cmd += f" --features_path {features_path[0]}"

    p = bash_command(cmd)
    p.wait()


def main(model_folder_cp,
         cp_folder,
         test_path,
         device,
         **kwargs):
    """
    Get predictions for all models.
    Args:
      model_folder_cp (str): directory in which all the model folders
        can be found
      cp_folder (str): path to the chemprop folder on your computer
      test_path (str): path to the file with the test SMILES and their properties
      device (Union[str, int]): device to evaluate the model on
    Returns:
      None
    """

    folders = os.listdir(model_folder_cp)
    # go through each folder

    for folder in folders:
        cp_model_path = os.path.join(model_folder_cp,
                                     folder)

        # continue if it's a file not a folder
        if not os.path.isdir(cp_model_path):
            continue

        # make predictions
        predict(cp_folder=cp_folder,
                test_path=test_path,
                cp_model_path=cp_model_path,
                device=device)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_folder_cp", type=str,
                        help=("Folder in which you will train your "
                              "ChemProp model. Models with different "
                              "parameters will get their own folders, "
                              "each located in `model_folder_cp`."))
    parser.add_argument("--cp_folder", type=str,
                        help=("Path to ChemProp folder."))
    parser.add_argument("--test_path", type=str,
                        help=("Path to the CSV with test set SMILES "
                              "and their actual property values"))
    parser.add_argument("--device", type=str,
                        help=("Device to use for model evaluation: "
                              "either the index of the GPU, "
                              "or 'cpu'. "))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    main(**args.__dict__)
