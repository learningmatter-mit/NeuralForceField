import sys
from tqdm import tqdm
import json
import subprocess
import os

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


def prepare_metric(lines, metric):

    header_items = [i.strip() for i in lines[0].split("|")]
    if metric in ["prc_auc", "prc-auc"]:
        metric = "pr_auc"
    elif metric in ["auc", "roc-auc"]:
        metric = "roc_auc"
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
