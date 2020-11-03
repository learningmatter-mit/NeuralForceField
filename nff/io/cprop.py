"""
Various functions to interface with chemprop.
"""

import json
import os
import numpy as np

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


def get_smiles(smiles_folder, name):
    """
    Get SMILES strings from csv.
    Args:
      smiles_folder (str): folder with the csvs
      name (str): csv file name
    Returns:
      smiles_list (list[str]): SMILES strings 
    """

    path = os.path.join(smiles_folder, name)
    with open(path, "r") as f:
        lines = f.readlines()
    smiles_list = [i.strip() for i in lines[1:]]
    return smiles_list


def save_smiles(smiles_folder, smiles_list, name):
    """
    Re-save the SMILES strings, ignoring those that aren't in
    `smiles_list`.

    Args:
      smiles_folder (str): folder with the csvs
      smiles_list (list[str]): SMILES strings that we will use
        in training -- excludes those that, for example, do not
        have 3D structures.
      name (str): csv file name

    Returns:
      None

    """

    # both the file with only the SMILES string, and the file
    # that has the SMILES string with its properties (e.g. bind /
    # no bind):

    file_names = [f"{name}_smiles.csv", f"{name}_full.csv"]
    paths = [os.path.join(smiles_folder, name) for name in
             file_names]
    for path in paths:
        with open(path, "r") as f:
            lines = f.readlines()
        keep_lines = [lines[0]]
        for line in lines[1:]:
            smiles = line.split(",")[0].strip()

            # keep the line only if the SMILES string is in
            # `smiles_list`

            if smiles in smiles_list:
                keep_lines.append(line)
        text = "".join(keep_lines)
        with open(path, "w") as f:
            f.write(text)


def make_hyperopt_csvs(smiles_folder, all_smiles):
    """
    Make csv files with SMILES strings for hyperparameter optimization.
    Args:
      smiles_folder (str): folder with the csvs
      all_smiles (list[str]): combined train and val SMILES for hyperparameter
        optimization that are actually used
    Returns:
      None
    """

    # full csv with properties, and just smiles csv
    suffixes = ["smiles", "full"]
    # dictionary with the combined lines read from train and val csvs
    # for each of the suffixes
    combined_lines = {suffix: [] for suffix in suffixes}

    for i, name in enumerate(["train", "val"]):
        for suffix in suffixes:
            file_path = os.path.join(smiles_folder, f"{name}_{suffix}.csv")
            with open(file_path, "r") as f:
                lines = f.readlines()

            # only include the header in the first file
            if i != 0:
                lines = lines[1:]
            combined_lines[suffix] += lines

    # write to new hyperopt csvs
    for suffix, lines in combined_lines.items():
        text = "".join(lines)
        new_path = os.path.join(smiles_folder, f"hyperopt_{suffix}.csv")
        with open(new_path, "w") as f:
            f.write(text)

    # re-save to account for the fact that not all smiles are used
    save_smiles(smiles_folder, all_smiles, name="hyperopt")


def save_hyperopt(feat_folder,
                  metric,
                  smiles_folder,
                  cp_save_folder,
                  dset_size):
    """
    Aggregate and save the train and validation SMILES for hyperparameter optimization.
    Args:
      feat_folder (str): path to the folder that contains all the feature files.
      metric (str): metric with which you're evaluating the model performance
      smiles_folder (str): folder with the csvs
      cp_save_folder (str): folder in which you're saving features for chemprop use
      dset_size (int, optional): maximum size of the entire dataset to use in hyperparameter 
        optimization.
    Returns:
      hyp_np_path (str): path of npz features file for hyperparameter optimization
    """

    names = ["train", "val"]
    all_feats = []
    all_smiles = []

    for name in names:
        smiles_list = get_smiles(smiles_folder, f"{name}_smiles.csv")
        np_save_path = os.path.join(cp_save_folder,
                                    f"{name}_{metric}.npz")
        feats = np.load(np_save_path)['features']
        all_feats.append(feats)
        all_smiles += smiles_list

    all_feats = np.concatenate(all_feats)

    if dset_size is not None:
        all_smiles = all_smiles[:dset_size]
        all_feats = all_feats[:dset_size]

    # save the entire train + val dataset features
    hyp_np_path = os.path.join(cp_save_folder,
                               f"hyperopt_{metric}.npz")
    np.savez_compressed(hyp_np_path, features=all_feats)

    # save csvs for the train + val dataset
    make_hyperopt_csvs(smiles_folder=smiles_folder,
                       all_smiles=all_smiles)

    return hyp_np_path
