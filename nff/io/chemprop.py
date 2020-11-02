"""
Various functions to interface with chemprop.
"""

import json
import os

from nff.utils import bash_command, fprint


def get_cp_cmd(script,
               config_path,
               data_path,
               dataset_type):
    """
    Get the string for a ChemProp command.
    Args:
      script (str): the path to the chemprop script you're running
      config_path (str): path to the config file for the job
      data_path (str): path to the dataset being used
      dataset_type (str): type of problem you're doing (e.g. regression,
        classification, multiclass)
    Returns:
      cmd (str): the chemprop command
    """

    cmd = (f"python {script} --config_path {config_path} "
           f" --data_path {data_path} "
           f" --dataset_type {dataset_type}")
    return cmd


def cp_hyperopt(cp_folder,
                hyp_folder,
                rerun):
    """
    Run hyperparameter optimization with ChemProp.
    Args:
      cp_folder (str): path to the chemprop folder on your computer
      hyp_folder (str): where you want to store your hyperparameter
        optimization models
      rerun (bool): whether to rerun hyperparameter optimization if
        `hyp_folder` already exists and has the completion file
        `best_params.json`.
    Returns:
      best_params (dict): best parameters from hyperparameter 
        optimization
    """

    # path to `best_params.json` file
    param_file = os.path.join(hyp_folder, "best_params.json")
    params_exist = os.path.isfile(param_file)

    # If it exists and you don't want to re-run, then load it
    if params_exist and (not rerun):

        fprint(f"Loading hyperparameter results from {param_file}\n")

        with open(param_file, "r") as f:
            best_params = json.load(f)
        return best_params

    # otherwise run the script and read in the results

    hyp_script = os.path.join(cp_folder, "hyperparameter_optimization.py")
    config_path = os.path.join(hyp_folder, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = config["data_path"]
    dataset_type = config["dataset_type"]
    cmd = get_cp_cmd(hyp_script,
                     config_path,
                     data_path,
                     dataset_type)
    cmd += f" --config_save_path {param_file}"

    fprint(f"Running hyperparameter optimization in folder {hyp_folder}\n")

    fprint(cmd)
    p = bash_command(f"source activate chemprop && {cmd}")
    p.wait()

    with open(param_file, "r") as f:
        best_params = json.load(f)

    return best_params


def cp_train(cp_folder,
             train_folder):
    """
    Train a chemprop model.
    Args:
      cp_folder (str): path to the chemprop folder on your computer
      train_folder (str): where you want to store your trained models
    Returns:
      None
    """

    train_script = os.path.join(cp_folder, "train.py")
    config_path = os.path.join(train_folder, "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = config["data_path"]
    dataset_type = config["dataset_type"]
    cmd = get_cp_cmd(train_script,
                     config_path,
                     data_path,
                     dataset_type)

    p = bash_command(f"source activate chemprop && {cmd}")
    p.wait()


def make_feat_paths(feat_path):
    """
    Make a feature path into a list.
    Args:
      feat_path (str): feature path
    Returns:
      paths (list): list of paths
    """

    if feat_path is not None:
        paths = [feat_path]
    else:
        paths = None
    return paths


def modify_config(base_config_path,
                  metric,
                  train_feat_path,
                  val_feat_path,
                  test_feat_path,
                  train_folder,
                  features_only,
                  hyp_params,
                  no_features):
    """
    Modify a chemprop config file with new parameters.
    Args:
      base_config_path (str): where your basic job config file
        is, with parameters that may or may not be changed depending
        on the given run
      metric (str): what metric you want to optimize in this run
      train_feat_path (str): where the features of your training set are
      val_feat_path (str): where the features of your validation set are
      test_feat_path (str): where the features of your test set are
      train_folder (str): where you want to store your trained models
      features_only (bool): whether to just train with the features and no
        MPNN
      hyp_params (dict): any hyperparameters that may have been optimized
      no_features (bool): Don't use external features when training model.
    Returns:
      None
    """

    with open(base_config_path, "r") as f:
        config = json.load(f)

    dic = {"metric": metric,
           "features_path": make_feat_paths(train_feat_path),
           "separate_val_features_path": make_feat_paths(val_feat_path),
           "separate_test_features_path": make_feat_paths(test_feat_path),
           "save_dir": train_folder,
           "features_only": features_only,
           **hyp_params}

    config.update({key: val for key, val in
                   dic.items() if val is not None})

    if no_features:
        for key in list(config.keys()):
            if "features_path" in key:
                config.pop(key)

    new_config_path = os.path.join(train_folder, "config.json")
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)


def modify_hyp_config(hyp_config_path,
                      metric,
                      hyp_feat_path,
                      hyp_folder,
                      features_only,
                      no_features):
    """
    Modfiy a hyperparameter optimization config file with new parameters.
    Args:
      hyp_config_path (str): where your basic hyperopt job config file
        is, with parameters that may or may not be changed depending
        on the given run
      metric (str): what metric you want to optimize in this run
      hyp_feat_path (str): path to all the features of the species that are
        part of the hyperparameter optimization (train and val from the
        real dataset).
      hyp_folder (str): where you want to store your trained models
      features_only (bool): whether to just train with the features and no
        MPNN
      no_features (bool): Don't use external features when training model.
    Returns:
      None
    """

    with open(hyp_config_path, "r") as f:
        config = json.load(f)

    dic = {"metric": metric,
           "features_path": make_feat_paths(hyp_feat_path),
           "save_dir": hyp_folder,
           "features_only": features_only}

    config.update({key: val for key, val in
                   dic.items() if val is not None})

    if no_features:
        for key in list(config.keys()):
            if "features_path" in key:
                config.pop(key)

    new_config_path = os.path.join(hyp_folder, "config.json")
    if not os.path.isdir(hyp_folder):
        os.makedirs(hyp_folder)

    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)
