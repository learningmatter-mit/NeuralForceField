"""
Get fingerprints produced by a 3D conformed-basedmodel.
"""

import json
import os
import subprocess
import pickle
import time
import argparse
from tqdm import tqdm

from nff.utils import (METRICS, CHEMPROP_TRANSFORM, parse_args, get_split_names)

# dictionary that transforms our metric syntax to chemprop's
REVERSE_TRANSFORM = {val: key for key, val in CHEMPROP_TRANSFORM.items()}
# available metrics
METRIC_LIST = [REVERSE_TRANSFORM.get(metric, metric) for metric in METRICS]


def make_base_config(config_file, kwargs, par):
    """
    Make a config file for `fps_single.py`.
    Args:
        config_file (str): path to general config file
        kwargs (dict): extra dictionary items to put in it
        par (bool): whether we're going to be using
            this to make fingerprints in parallel.
    Returns:
        base_path (str): path to the new, base config
            file, which is missing certain keys if we're
            running this in parallel.
    """

    # get the contents of the config file
    with open(config_file, "r") as f_open:
        config = json.load(f_open)

    # update it with any new values
    for key, val in kwargs.items():
        # if you put the config file in the dictionary,
        # but the config file is the original name and
        # not the new name, it messes things up
        if key not in config and key != "config_file":
            config[key] = val

    # if running in parallel, get rid of these keys because
    # we'll want to specify them individually for each parallel
    # process

    if par:
        par_keys = ["dset_folder", "feat_save_folder", "config_file",
                    "slurm_parallel"]
        for key in par_keys:
            if key in config:
                config.pop(key)

    # make a new path to this "base" config and save
    base_path = config_file.replace(".json", "_base.json")
    with open(base_path, "w") as f_open:
        json.dump(config, f_open, indent=4, sort_keys=True)
    return base_path


def get_single_path():
    """
    Get the path to `fps_single.py`.
    Args:
        None
    Returns:
        single_path (str): path to `fps_single.py`
    """
    this_dir = os.path.abspath(".")
    single_path = os.path.join(this_dir, "fps_single.py")
    return single_path


def run_par(base_config_file,
            dset_folder,
            idx):
    """
    Make fingerprints in parallel.
    Args:
        base_config_file (str): path to the new, base config
            file, which is missing certain keys if we're
            running this in parallel.
        dset_folder (str): path to dataset
        idx (str): index of the dataset thread we're working
            on.
    Returns:
        p: subprocess from executing the parallel command
    """

    # create the command
    single_path = get_single_path()
    idx_folder = os.path.join(dset_folder, str(idx))
    cmd = (f"srun -N 1 -n 1 --exclusive python {single_path} "
           f" --config_file {base_config_file} "
           f" --dset_folder {idx_folder} --feat_save_folder {idx_folder} ")

    # figure out whether we should be tracking progress of this thread
    # or not
    num_nodes = os.environ["SLURM_NNODES"]
    if (int(idx) % int(num_nodes) != 0):
        cmd += "--no_track"

    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)
    return p


def monitor_results(dset_folder,
                    folders,
                    split_names,
                    metric):
    """
    Monitor results of parallel processes using tqdm.
    Args:
        dset_folder (str): path to dataset
        folders (list[str]): names of sub-folders
        split_names (list[str]): names of splits
            (train, val, and/or test) that we're 
            monitoring.
        metric (str): name of the metric that we're using
            to evaluate the model.
    Returns:
        None
    """

    # Make a dictionary that has the path of each pickle
    # file we're going to make for each sub-folder. Initialize
    # each value to False

    monitor_dic = {}
    for split in split_names:
        for folder in folders:
            pickle_name = f"pred_{metric}_{split}.pickle"
            pickle_path = os.path.join(dset_folder, folder, pickle_name)
            monitor_dic[pickle_path] = False

    # Update the dictionary as those files are made and use it to
    # update the tqdm progress bar. Keep looping until all the files
    # exist

    total = len(monitor_dic)
    with tqdm(total=total) as pbar:
        while False in monitor_dic.values():
            for path, val in monitor_dic.items():
                if os.path.isfile(path) and not val:
                    monitor_dic[path] = True
                    pbar.update(1)
            time.sleep(5)


def pickle_sub_path(metric,
                    split,
                    folder,
                    dset_folder):
    """
    Get path to the pickle file in a dataset sub-folder.
    Args:
        metric (str): name of the metric that we're using
            to evaluate the model.
        split (str): name of split (train, val or test)
        folder (str): name of sub-folder
        dset_folder (str): path to dataset
    Returns:
        path (str): path to pickle file
    """
    pickle_name = f"pred_{metric}_{split}.pickle"
    path = os.path.join(dset_folder, folder, pickle_name)

    return path


def combine_results(dset_folder,
                    metric,
                    split_names):
    """
    Combine results from different parallel processes into one big 
    dictionary.
    Args:
        dset_folder (str): path to dataset
        metric (str): name of the metric that we're using
            to evaluate the model.
        split_names (list[str]): names of splits
            (train, val, and/or test) that we're 
            monitoring.
    Returns:
        combined_dics (dict): dictionary of the form {split: sub_dic}
            for split in each dataset split (train, val, and/or test),
            and sub_dic is the results dictionary (contains predicted
            and actual quantity values, fingeprints, etc.)
    """

    # find the folders and track results
    folders = sorted([i for i in os.listdir(dset_folder) if i.isdigit()],
                     key=lambda x: int(x))
    monitor_results(dset_folder, folders, split_names, metric)

    # once all the results are in, put them into a big dictionary
    combined_dics = {}

    # go through each split
    for split in split_names:
        overall = {}
        # go through each folder and loop until you've succesfully loaded
        # all pickles
        for folder in folders:
            while True:
                pickle_path = pickle_sub_path(
                    metric, split, folder, dset_folder)
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

def run_all_par(kwargs):
    """
    Run all the parallel processes.
    Args:
        kwargs (dict): dictionary of keywords
    Retuns:
        None
    """

    dset_folder = kwargs["dset_folder"]
    config_file = kwargs["config_file"]
    metric = kwargs["metric"]
    feat_save_folder = kwargs["feat_save_folder"]

    # make the config file that has the basic parameters, but
    # is missing others that will depend on the process being used
    base_config_file = make_base_config(config_file=config_file,
                                        kwargs=kwargs,
                                        par=True)

    # get the dataset folders
    folders = sorted([i for i in os.listdir(dset_folder) if i.isdigit()],
                     key=lambda x: int(x))
    procs = []

    split_names = get_split_names(train_only=kwargs.get("train_only"),
                                  val_only=kwargs.get("val_only"),
                                  test_only=kwargs.get("test_only"))
    # submit the parallel command
    for idx in folders:
        paths = [pickle_sub_path(metric, split, idx, dset_folder)
                 for split in split_names]
        if all([os.path.isfile(path) for path in paths]):
            continue
        p = run_par(base_config_file, dset_folder, idx)
        procs.append(p)

    # get the final results
    results = combine_results(dset_folder, metric, split_names)

    # save them in the feature folder as a pickle file
    for split, sub_dic in results.items():
        pickle_name = f"pred_{metric}_{split}.pickle"
        pickle_path = os.path.join(feat_save_folder, pickle_name)
        with open(pickle_path, "wb") as f:
            pickle.dump(sub_dic, f)


def run_single(kwargs):
    """
    Make fingerprints in series.
    Args:
        kwargs (dict): dictionary of keywords
    Retuns:
        None
    """
    config_file = kwargs["config_file"]
    metric = kwargs["metric"]

    # save the arguments in a config file so we can just specify
    # its path in the command, instead of adding them as command
    # line arguments
    base_config_file = make_base_config(config_file=config_file,
                                        kwargs=kwargs,
                                        par=False)
    # get the path to `fps_single.py
    single_path = get_single_path()
    # execute the command
    cmd = (f"python {single_path} --config_file {base_config_file}")
    print(cmd)
    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)
    p.wait()

def main(kwargs):
    """
    Get the fingerprints and results from the model.
    Args:
        kwargs (dict): dictionary of keywords
    Retuns:
        None
    """
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
    parser.add_argument('--train_only', action='store_true',
                        help=("Only evaluate model "
                              "and generate fingerprints for "
                              "the training set"))
    parser.add_argument('--val_only', action='store_true',
                        help=("Only evaluate model "
                              "and generate fingerprints for "
                              "the validation set"))
    parser.add_argument('--slurm_parallel', action='store_true',
                        help=("Use slurm to evaluate model predictions "
                              "in parallel over different nodes."))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments."))

    args = parse_args(parser)
    kwargs = args.__dict__

    main(kwargs)

