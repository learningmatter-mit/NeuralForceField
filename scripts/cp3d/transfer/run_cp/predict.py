"""
Wrapper to get the ChemProp predictions on the test set for each model in
the folder `model_folder_cp`.
"""

import os
import json
import argparse

from nff.utils import bash_cmd, parse_args


def predict(cp_path,
            test_path,
            cp_model_path,
            gpu):

    script = os.path.join(cp_path, "predict.py")
    preds_path = os.path.join(cp_model_path, f"test_pred.csv")

    check_names = [i for i in os.listdir(cp_model_path)
                   if i.startswith("fold_") and i.split("_")[-1].isdigit()]
    check_paths = [os.path.join(cp_model_path, name, "model_0")
                   for name in check_names]

    # then not a model path
    if len(check_paths) == 0:
        return

    args_path = f"{cp_model_path}/fold_0/args.json"
    with open(args_path, "r") as f:
        args = json.load(f)
    features_path = args["separate_test_features_path"]

    check_str = " ".join(check_paths)

    cmd = (f"python {script} "
           f" --test_path {test_path} --preds_path {preds_path} "
           f" --checkpoint_paths {check_str} "
           f" --gpu {gpu} ")

    if features_path is not None:
        cmd += f" --features_path {features_path[0]}"

    bash_cmd(cmd)


def main(model_folder_cp,
         cp_path,
         test_path,
         gpu):

    folders = os.listdir(model_folder_cp)
    for folder in folders:
        cp_model_path = os.path.join(model_folder_cp,
                                     folder)

        predict(cp_path=cp_path,
                test_path=test_path,
                cp_model_path=cp_model_path,
                gpu=gpu)


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
    parser.add_argument("--gpu", type=int,
                        help=("Index of the GPU to use for evaluating "
                              "the model."))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    main(**args.__dict__)
