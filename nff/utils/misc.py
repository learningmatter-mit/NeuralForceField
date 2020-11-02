import sys
from tqdm import tqdm
import json
import subprocess
import os
import random

METRIC_DIC = {"pr_auc": "maximize",
              "roc_auc": "maximize",
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
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def parse_args(parser, config_flag="config_file"):
    args = parser.parse_args()

    config_path = getattr(args, config_flag, None)
    if config_path is not None:
        with open(config_path, "r") as f:
            config_args = json.load(f)
        for key, val in config_args.items():
            if hasattr(args, key):
                setattr(args, key, val)
    return args


def fprint(msg):
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
    if metric in ["prc_auc", "prc-auc"]:
        metric = "pr_auc"
    elif metric in ["auc", "roc-auc"]:
        metric = "roc_auc"
    return metric


def prepare_metric(lines, metric):

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
