"""
Wrapper to get the ChemProp predictions on the test set for each model in
the folder `model_folder_cp`.
"""

import os
import json
import argparse
import numpy as np

from nff.utils import (bash_command, parse_args, read_csv,
                       fprint, CHEMPROP_METRICS, apply_metric)


def is_model_path(cp_model_path):
    """
    Check to see if a directory is actually a model path.
    Args:
      cp_model_path (str): path to folder
    Returns:
      check_paths (list[str]): paths to the different model checkpoints
      is_model (bool): whether it's really a model path
    """

    # get the paths of all the models saved with different initial random seeds
    check_names = [i for i in os.listdir(cp_model_path)
                   if i.startswith("fold_") and i.split("_")[-1].isdigit()]
    # sort by order
    check_names = sorted(check_names, key=lambda x: int(x.split("_")[-1]))
    check_paths = [os.path.join(cp_model_path, name, "model_0/model.pt")
                   for name in check_names]
    is_model = len(check_paths) != 0

    return check_paths, is_model


def predict(cp_folder,
            test_path,
            cp_model_path,
            device,
            check_paths):
    """
    Get and save the prediction results from a ChemProp model.
    Args:
      cp_folder (str): path to the chemprop folder on your computer
      test_path (str): path to the file with the test SMILES and their properties
      cp_model_path (str): path to the folder with the model of interest
      device (Union[str, int]): device to evaluate the model on
      check_paths (list[str]): paths to the different model checkpoints
    Returns:
      reals (dict):dictionary of the form {prop: real}, where `real`
          are the real values of the property `prop`.
      preds (list[dict]): same as `real` but for predicted. One for each
          model.
    """

    script = os.path.join(cp_folder, "predict.py")
    preds_path = os.path.join(cp_model_path, f"test_pred.csv")

    # load the arguments from that model to get the features path
    args_path = f"{cp_model_path}/fold_0/args.json"
    if not os.path.isfile(args_path):
        args_path = args_path.replace("fold_0/", "")
    with open(args_path, "r") as f:
        args = json.load(f)
    features_path = args["separate_test_features_path"]

    # predictions from different models
    preds = []

    for i, check_path in enumerate(check_paths):

        # make the chemprop command

        this_path = preds_path.replace(".csv", f"_{i}.csv")
        cmd = (f"source activate chemprop && python {script} "
               f" --test_path {test_path} --preds_path {this_path} "
               f" --checkpoint_paths {check_path} ")

        if device == "cpu":
            cmd += f" --no_cuda"
        else:
            cmd += f" --gpu {device} "

        if features_path is not None:
            feat_str = " ".join(features_path)
            cmd += f" --features_path {feat_str}"

        p = bash_command(cmd)
        p.wait()

        pred = read_csv(this_path)
        preds.append(pred)

    real = read_csv(test_path)

    return real, preds


def get_metrics(actual_dic, pred_dics, metrics, cp_model_path):
    """
    Get all requested metric scores for a set of predictions and save
    to a JSON file.
    Args:
      actual_dic (dict): dictionary of the form {prop: real}, where `real` are the
        real values of the property `prop`.
      pred_dics (list[dict]): list of dictionaries, each the same as `real` but
        with values predicted by each different model.
      metrics (list[str]): metrics to apply
      cp_model_path (str): path to the folder with the model of interest
    Returns:
      None
    """

    overall_dic = {}
    for i, pred_dic in enumerate(pred_dics):
        metric_dic = {}
        for prop in pred_dic.keys():
            if prop == "smiles":
                continue
            actual = actual_dic[prop]
            pred = pred_dic[prop]

            metric_dic[prop] = {}

            for metric in metrics:
                score = apply_metric(metric, pred, actual)
                metric_dic[prop][metric] = score

            overall_dic[str(i)] = metric_dic

    props = [prop for prop in pred_dic.keys() if prop != 'smiles']
    overall_dic['average'] = {prop: {} for prop in props}
    sub_dics = [val for key, val in overall_dic.items() if key != 'average']

    for prop in props:
        for key in sub_dics[0][prop].keys():
            vals = [sub_dic[prop][key] for sub_dic in sub_dics]
            mean = np.mean(vals).item()
            std = np.std(vals).item()
            overall_dic['average'][prop][key] = {"mean": mean, "std": std}

    save_path = os.path.join(cp_model_path, f"test_metrics.json")
    with open(save_path, "w") as f:
        json.dump(overall_dic, f, indent=4, sort_keys=True)

    fprint(f"Saved metric scores to {save_path}")


def main(model_folder_cp,
         cp_folder,
         test_path,
         device,
         metrics,
         **kwargs):
    """
    Get predictions for all models and evaluate with a set of metrics.
    Args:
      model_folder_cp (str): directory in which all the model folders
        can be found
      cp_folder (str): path to the chemprop folder on your computer
      test_path (str): path to the file with the test SMILES and their properties
      device (Union[str, int]): device to evaluate the model on
      metrics (list[str]): metrics to apply 
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

        check_paths, is_model = is_model_path(cp_model_path)
        if not is_model:
            continue

        # make predictions
        real, preds = predict(cp_folder=cp_folder,
                              test_path=test_path,
                              cp_model_path=cp_model_path,
                              device=device,
                              check_paths=check_paths)

        # get and save metric scores
        get_metrics(real, preds, metrics, cp_model_path)


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
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=None, choices=CHEMPROP_METRICS,
                        help=("Optional metrics with which you want to "
                              "evaluate predictions."))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    main(**args.__dict__)
