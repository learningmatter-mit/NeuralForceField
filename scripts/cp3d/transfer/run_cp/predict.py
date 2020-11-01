"""
Wrapper to get the ChemProp predictions on the test set for each model in
the folder `model_folder_cp`.
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import (roc_auc_score, auc, precision_recall_curve,
                             r2_score, accuracy_score, log_loss)

from nff.utils import (bash_command, parse_args, read_csv,
                       fprint, CHEMPROP_METRICS)


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
      real (dict): dictionary of the form {prop: real}, where `real` are the real
        values of the property `prop`.
      pred (dict): same as `real` but for predicted.
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

    real = read_csv(test_path)
    pred = read_csv(preds_path)

    return real, pred


def apply_metric(metric, pred, actual):
    """
    Apply a metric to a set of predictions.
    Args:
      metric (str): name of metric
      pred (iterable): predicted values
      actual (iterable): actual values
    Returns:
      score (float): metric score
    """
    if metric == "auc":
        score = roc_auc_score(y_true=actual, y_score=pred)
    elif metric == "prc-auc":
        precision, recall, thresholds = precision_recall_curve(
            y_true=actual, probas_pred=pred)
        score = auc(recall, precision)
    elif metric == "mse":
        score = ((np.array(pred) - np.array(actual)) ** 2).mean()
    elif metric == "rmse":
        score = ((np.array(pred) - np.array(actual)) ** 2).mean() ** 0.5
    elif metric == "mae":
        score = (abs(np.array(pred) - np.array(actual))).mean()
    elif metric == "r2":
        score = r2_score(y_true=actual, y_pred=pred)
    elif metric == "accuracy":
        np_pred = np.array(pred)
        mask = np_pred >= 0.5
        np_pred[mask] = 1
        np_pred[np.bitwise_not(mask)] = 0
        score = accuracy_score(y_true=actual, y_pred=np_pred)
    elif metric in ["cross_entropy", "binary_cross_entropy"]:
        score = log_loss(y_true=actual, y_pred=np_pred)

    return score


def get_metrics(actual_dic, pred_dic, metrics, cp_model_path):
    """
    Get all requested metric scores for a set of predictions and save
    to a JSON file.
    Args:
      actual_dic (dict): dictionary of the form {prop: real}, where `real` are the 
        real values of the property `prop`.
      pred_dic (dict): same as `real` but for predicted.
      metrics (list[str]): metrics to apply 
      cp_model_path (str): path to the folder with the model of interest
    Returns:
      None
    """

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

    save_path = os.path.join(cp_model_path, f"test_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metric_dic, f, indent=4, sort_keys=True)

    fprint(f"Saved metric score to {save_path}")


def main(model_folder_cp,
         cp_folder,
         test_path,
         device,
         metrics,
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

        check_paths, is_model = is_model_path(cp_model_path)
        if not is_model:
            continue

        # make predictions
        real, pred = predict(cp_folder=cp_folder,
                             test_path=test_path,
                             cp_model_path=cp_model_path,
                             device=device,
                             check_paths=check_paths)

        # get and save metric scores
        get_metrics(real, pred, metrics, cp_model_path)


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
