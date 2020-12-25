import sys
from tqdm import tqdm
import json
import subprocess
import os
import random
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, auc, precision_recall_curve,
                             r2_score, accuracy_score, log_loss)

# optimization goal for various metrics

METRIC_DIC = {"pr_auc": "maximize",
              "roc_auc": "maximize",
              "r2": "maximize",
              "class_loss": "minimize",
              "regress_loss": "minimize",
              "mae": "minimize",
              "mse": "minimize"}

METRICS = list(METRIC_DIC.keys())

# transform from chemprop syntax to our syntax for the metrics

CHEMPROP_TRANSFORM = {"auc": "roc_auc",
                      "prc-auc": "pr_auc",
                      "binary_cross_entropy": "class_loss",
                      "mse": "regress_loss"}

# metrics available in chemprop

CHEMPROP_METRICS = ["auc",
                    "prc-auc",
                    "rmse",
                    "mae",
                    "mse",
                    "r2",
                    "accuracy",
                    "cross_entropy",
                    "binary_cross_entropy"]


def tqdm_enum(iter):
    """
    Wrap tqdm around `enumerate`.
    Args:
        iter (iterable): an iterable (e.g. list)
    Returns
        i (int): current index
        y: current value
    """

    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def parse_args(parser, config_flag="config_file"):
    """
    Parse arguments.
    Args:
        parser (argparse.ArgumentParser): argument parser
        config_flag (str): name of the arg key
            that gives the name of the config file.
    Returns:
        args (argparse.Namespace): arguments
    """

    # parse the arguments
    args = parser.parse_args()

    # if the config path is specified, then load
    # arguments from that file and apply the results
    # to `args`

    config_path = getattr(args, config_flag, None)
    if config_path is not None:
        with open(config_path, "r") as f:
            config_args = json.load(f)
        for key, val in config_args.items():
            if hasattr(args, key):
                setattr(args, key, val)
    return args


def fprint(msg):
    """
    Print a string immediately.
    Args:
        msg (str): string to print
    Returns:
        None
    """
    print(msg)
    sys.stdout.flush()


def bash_command(cmd):
    """ Run a command from the command line using subprocess.
    Args:
        cmd (str): command
    Returns:
        None
    """

    return subprocess.Popen(cmd, shell=True, executable='/bin/bash')


def convert_metric(metric):
    """
    Convert a metric name to a fixed name that can be used in 
    various scripts.
    Args:
        metric (str): input metric
    Returns:
        metric (str): output metric
    """
    if metric in ["prc_auc", "prc-auc"]:
        metric = "pr_auc"
    elif metric in ["auc", "roc-auc"]:
        metric = "roc_auc"
    return metric


def prepare_metric(lines, metric):
    """
    Get various metric quantities before parsing a log fine.
    Args:
        lines (list[str]): lines in the log file
        metric (str): name of metric
    Returns:
        idx (int): index at which the metric score occurs
            when the given line has been split by `|`
        best_score (float): initial best score
        best_epoch (int): initial best_epoch
        optim (str): goal of the metric optimization (i.e.
            minimize or maximize.)
    """
    header_items = [i.strip() for i in lines[0].split("|")]
    metric = convert_metric(metric)
    if "loss" in metric:
        idx = header_items.index("Validation loss")
    else:
        for i, item in enumerate(header_items):
            sub_keys = metric.split("_")
            if all([key.lower() in item.lower()
                    for key in sub_keys]):
                idx = i

    optim = METRIC_DIC[metric]

    if optim == "minimize":
        best_score = float("inf")
    else:
        best_score = -float("inf")

    best_epoch = -1

    return idx, best_score, best_epoch, optim


def parse_score(model_path, metric):
    """
    Find the best score and best epoch according to a given metric.
    Args:
        model_path (str): path to the training folder
        metric (str): name of metric
    Returns:
        best_score (float): best validation score
        best_epoch (int): epoch with the best validation score
    """

    log_path = os.path.join(model_path, "log_human_read.csv")
    with open(log_path, "r") as f:
        lines = f.readlines()

    idx, best_score, best_epoch, optim = prepare_metric(
        lines=lines,
        metric=metric)

    for line in lines:
        splits = [i.strip() for i in line.split("|")]
        try:
            score = float(splits[idx])
        except ValueError:
            continue
        if any([(optim == "minimize" and score < best_score),
                (optim == "maximize" and score > best_score)]):
            best_score = score
            best_epoch = splits[1]

    return best_score, best_epoch


def read_csv(path):
    """
    Read a csv into a dictionary.
    Args:
        path (str): path to the csv file
    Returns:
        dic (dict): dictionary version of the file
    """
    with open(path, "r") as f:
        lines = f.readlines()

    keys = lines[0].strip().split(",")
    dic = {key: [] for key in keys}
    for line in lines[1:]:
        vals = line.strip().split(",")
        for key, val in zip(keys, vals):
            if val.isdigit():
                dic[key].append(int(val))
            else:
                try:
                    dic[key].append(float(val))
                except ValueError:
                    dic[key].append(val)
    return dic


def write_csv(path, dic):
    """
    Write a dictionary to a csv.
    Args:
        path (str): path to the csv file
        dic (dict): dictionary
    Returns:
        None
    """

    keys = sorted(list(dic.keys()))
    if "smiles" in keys:
        keys.remove("smiles")
        keys.insert(0, "smiles")

    lines = [",".join(keys)]
    for idx in range(len(dic[keys[0]])):
        vals = [dic[key][idx] for key in keys]
        line = ",".join(str(val) for val in vals)
        lines.append(line)
    text = "\n".join(lines)

    with open(path, "w") as f:
        f.write(text)


def prop_split(max_specs,
               dataset_type,
               props,
               sample_dic,
               seed):
    """
    Sample a set of smiles strings by up to a maximum number. If the
    property of interest is a binary value, try to get as many of the
    underrepresented class as possible.
    Args:
        max_specs (int): maximum number of species
        dataset_type (str): type of problem (classification or regression)
        props (list[str]): names of properties you'll be fitting
        sample_dic (dict): dictionary of the form {smiles: sub_dic} for the
            set of smiles strings, where sub_dic contains other information,
            e.g. about `props`.
        seed (int): random seed for sampling
    Returns:
        keep_smiles (list[str]): sampled smiles strings.
    """

    random.seed(seed)

    if max_specs is not None and dataset_type == "classification":

        msg = "Not implemented for multiclass"
        assert len(props) == 1, msg

        prop = props[0]
        pos_smiles = [key for key, sub_dic in sample_dic.items()
                      if sub_dic.get(prop) == 1]
        neg_smiles = [key for key, sub_dic in sample_dic.items()
                      if sub_dic.get(prop) == 0]

        # find the underrepresnted and overrepresented class
        if len(pos_smiles) < len(neg_smiles):
            underrep = pos_smiles
            overrep = neg_smiles
        else:
            underrep = neg_smiles
            overrep = pos_smiles

        # if possible, keep all of the underrepresented class
        if max_specs >= 2 * len(underrep):
            random.shuffle(overrep)
            num_left = max_specs - len(underrep)
            keep_smiles = underrep + overrep[:num_left]

        # otherwise create a dataset with half of each
        else:
            random.shuffle(underrep)
            random.shuffle(overrep)
            keep_smiles = (underrep[:max_specs // 2]
                           + overrep[max_specs // 2:])
    else:

        keep_smiles = list(sample_dic.keys())

        # if setting a maximum, need to shuffle in order
        # to take random smiles

        if max_specs is not None:
            random.shuffle(keep_smiles)

    if max_specs is not None:
        keep_smiles = keep_smiles[:max_specs]

    return keep_smiles


def get_split_names(train_only,
                    val_only,
                    test_only):
    """
    Get names of dataset splits.
    Args:
      train_only (bool): only load the training set
      val_only (bool): only load the validation set
      test_only (bool): only load the test set
    Returns:
        names (list[str]): names of splits
            (train, val, and/or test) that we're
            monitoring.
    """

    only_dic = {"train": train_only,
                "val": val_only,
                "test": test_only}

    requested = [name for name, only in only_dic.items()
                 if only]
    if len(requested) > 1:
        string = ", ".join(requested)
        msg = (f"Requested {string}, which are mutually exclusive")
        raise Exception(msg)

    if len(requested) != 0:
        names = requested
    else:
        names = ["train", "val", "test"]

    return names


def preprocess_class(pred):
    """
    Preprocess classifier predictions. This applies,
    for example, if you train an sklearn regressor
    rather than classifier, which doesn't necessarily
    predict a value between 0 and 1.
    Args:
        pred (np.array or torch.Tensor or list): predictions
    Returns:
        pred (np.array or torch.Tensor or list): predictions
            with max 1 and min 0.
    """

    to_list = False
    if type(pred) is list:
        pred = np.array(pred)
        to_list = True

    # make sure the min and max are 0 and 1
    pred[pred < 0] = 0
    pred[pred > 1] = 1

    if to_list:
        pred = pred.tolist()

    return pred


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
        pred = preprocess_class(pred)
        if max(pred) == 0:
            score = 0
        else:
            score = roc_auc_score(y_true=actual, y_score=pred)
    elif metric == "prc-auc":
        pred = preprocess_class(pred)
        if max(pred) == 0:
            score = 0
        else:
            precision, recall, _ = precision_recall_curve(
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


def avg_distances(dset):
    """
    Args:
        dset (nff.nn.data.Dataset): NFF dataset where all the geometries are
            different conformers for one species.
    """

    # Get the neighbor list that includes the neighbor list of each conformer

    all_nbrs = []
    for nbrs in dset.props['nbr_list']:
        for pair in nbrs:
            all_nbrs.append(tuple(pair.tolist()))
    all_nbrs_tuple = list(set(tuple(all_nbrs)))

    all_nbrs = torch.LongTensor([list(i) for i in all_nbrs_tuple])

    num_confs = len(dset)
    all_distances = torch.zeros(num_confs, all_nbrs.shape[0])

    for i, batch in enumerate(dset):
        xyz = batch["nxyz"][:, 1:]
        all_distances[i] = ((xyz[all_nbrs[:, 0]] - xyz[all_nbrs[:, 1]])
                            .pow(2).sum(1).sqrt())

    weights = dset.props["weights"].reshape(-1, 1)
    avg_d = (all_distances * weights).sum(0)

    return all_nbrs, avg_d

