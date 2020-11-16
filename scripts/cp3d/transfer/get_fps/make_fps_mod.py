import json
import os
import subprocess
import pickle
import time
import numpy as np
import argparse
from tqdm import tqdm

from nff.utils import (METRICS, CHEMPROP_TRANSFORM, parse_args)

# dictionary that transforms our metric syntax to chemprop's
REVERSE_TRANSFORM = {val: key for key, val in CHEMPROP_TRANSFORM.items()}
# available metrics
METRIC_LIST = [REVERSE_TRANSFORM.get(metric, metric) for metric in METRICS]


def make_base_config(config_file, kwargs, par):

    with open(config_file, "r") as f_open:
        config = json.load(f_open)
    for key, val in kwargs.items():
        if key not in config:
            config[key] = val

    if par:
        par_keys = ["dset_folder", "feat_save_folder", "config_file",
                    "slurm_parallel"]
        for key in par_keys:
            if key in config:
                config.pop(key)

    base_path = config_file.replace(".json", "_base.json")
    with open(base_path, "w") as f_open:
        json.dump(config, f_open, indent=4, sort_keys=True)
    return base_path


def get_single_path():
    this_dir = os.path.abspath(".")
    single_path = os.path.join(this_dir, "fps_single.py")
    return single_path


def run_par(base_config_file,
            dset_folder,
            idx):

    num_nodes = os.environ["SLURM_NNODES"]
    single_path = get_single_path()
    idx_folder = os.path.join(dset_folder, str(idx))
    cmd = (f"srun -N 1 -n 1 python {single_path} --config_file {base_config_file} "
           f" --dset_folder {idx_folder} --feat_save_folder {idx_folder} ")

    if (int(idx) % int(num_nodes) != 0):
        cmd += "--no_track"

    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)
    return p

def monitor_results(dset_folder, folders, split_names, metric):
    monitor_dic = {}
    for split in split_names:
        for folder in folders:
            pickle_name = f"pred_{metric}_{split}.pickle"
            pickle_path = os.path.join(dset_folder, folder, pickle_name)
            monitor_dic[pickle_path] = False
    total = len(monitor_dic)
    with tqdm(total=total) as pbar:
        while False in monitor_dic.values():
            for path in monitor_dic.keys():
                if os.path.isfile(path) and not monitor_dic[path]:
                    monitor_dic[path] = True
                    pbar.update(1)
            time.sleep(5)

def combine_results(dset_folder, metric, split_names):

    folders = sorted([i for i in os.listdir(dset_folder) if i.isdigit()],
                     key=lambda x: int(x))
    monitor_results(dset_folder, folders, split_names, metric)

    combined_dics = {}

    for split in split_names:
        pickle_name = f"pred_{metric}_{split}.pickle"
        overall = {}
        for folder in folders:
            while True:
                pickle_path = os.path.join(dset_folder, folder, pickle_name)
                try:
                    with open(pickle_path, "rb") as f:
                        results = pickle.load(f)
                except (EOFError, FileNotFoundError, pickle.UnpicklingError):
                    time.sleep(1)
                    continue
                for key, val in results.items():
                    overall[key] = val
                break
        combined_dics[split] = overall
    return combined_dics

def get_split_names(kwargs):
    if kwargs.get("test_only"):
        names = ["test"]
    else:
        names = ["train", "val", "test"]
    return names

def run_all_par(kwargs):

    dset_folder = kwargs["dset_folder"]
    config_file = kwargs["config_file"]
    metric = kwargs["metric"]
    feat_save_folder = kwargs["feat_save_folder"]

    base_config_file = make_base_config(config_file=config_file,
                                        kwargs=kwargs,
                                        par=True)

    folders = sorted([i for i in os.listdir(dset_folder) if i.isdigit()],
                     key=lambda x: int(x))
    procs = []

    for idx in folders:
        p = run_par(base_config_file, dset_folder, idx)
        procs.append(p)

    split_names = get_split_names(kwargs)
    results = combine_results(dset_folder, metric, split_names)

    for split, sub_dic in results.items():
        pickle_name = f"pred_{metric}_{split}.pickle"
        pickle_path = os.path.join(feat_save_folder, pickle_name)
        with open(pickle_path, "wb") as f:
            pickle.dump(sub_dic, f)


def run_single(kwargs):
    config_file = kwargs["config_file"]
    base_config_file = make_base_config(config_file=config_file,
                                        kwargs=kwargs,
                                        par=False)
    single_path = get_single_path()
    cmd = f"python {single_path} --config_file {base_config_file}"
    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)
    p.wait()


def main(kwargs):

    slurm_parallel = kwargs["slurm_parallel"]

    if slurm_parallel:

        run_all_par(kwargs)
    else:
        run_single(kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str,
                        help="Name of model path")
    parser.add_argument('--dset_folder', type=str,
                        help=("Name of the folder with the "
                              "datasets you want to add "
                              "fingerprints to"))
    parser.add_argument('--feat_save_folder', type=str,
                        help="Path to save pickles")
    parser.add_argument('--device', type=str,
                        help="Name of device to use")
    parser.add_argument('--batch_size', type=int,
                        help="Batch size")
    parser.add_argument('--prop', type=str,
                        help="Property to predict",
                        default=None)
    parser.add_argument('--sub_batch_size', type=int,
                        help="Sub batch size",
                        default=None)
    parser.add_argument('--metric', type=str,
                        help=("Select the model with the best validation "
                              "score on this metric. If no metric "
                              "is given, the metric used in the training "
                              "process will be used."),
                        default=None,
                        choices=METRIC_LIST)
    parser.add_argument('--test_only', action='store_true',
                        help=("Only evaluate model "
                              "and generate fingerprints for "
                              "the test set"))
    parser.add_argument('--slurm_parallel', action='store_true',
                        help=("Use slurm to evaluate model predictions "
                              "in parallel over different nodes."))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments."))

    args = parse_args(parser)
    kwargs = args.__dict__

    main(kwargs)
