"""
A python wrapper to train a ChemProp model with optimized hyperparameters.
"""


import argparse
import json
import os

from nff.utils import (parse_args, CHEMPROP_METRICS,
                       prop_split, read_csv, write_csv)
from nff.io.cprop import (cp_train, cp_hyperopt,
                          modify_config, modify_hyp_config)


def make_hyp_csvs(base_config_path,
                  max_specs,
                  seed):
    """
    Make csv files for the subsection of the SMILES that will be used
    for hyperparameter optimization.
    Args:
      base_config_path (str): where your basic job config file
        is, with parameters that may or may not be changed depending
        on the given run
      max_specs (int): maximum number of species to use in hyperparameter
        optimization.
      seed (int): random seed to use for split.
    Returns:
      None
    """

    # load the base config
    with open(base_config_path, "r") as f:
        base_dic = json.load(f)

    # load the SMILES strings from the train and validation
    # paths, then sample them
    train_path = base_dic["data_path"]
    val_path = base_dic.get("separate_val_path")
    paths = [train_path, val_path]

    # initialize the dictionary by reading the train data
    prop_dic = read_csv(paths[0])

    # if the validation data is separate, add the data lists
    # together

    if val_path is not None:
        new_dic = read_csv(val_path)
        for key, val in new_dic.items():
            prop_dic[key] += val

    # generate a proportional sample by first getting the
    # properties to be predicted, then making a `sample_dic`,
    # and finally calling `prop_split`

    props = list(filter(lambda x: x != "smiles",
                        prop_dic.keys()))
    dataset_type = base_dic.get("dataset_type", "regression")

    num_smiles = len(prop_dic["smiles"])
    sample_dic = {prop_dic["smiles"][idx]: {prop: prop_dic[prop][idx]
                                            for prop in props}
                  for idx in range(num_smiles)}

    keep_smiles = prop_split(max_specs=max_specs,
                             dataset_type=dataset_type,
                             props=props,
                             sample_dic=sample_dic,
                             seed=seed)

    # save to csv
    new_dic = {"smiles": keep_smiles}
    for prop in props:
        new_dic.update({prop: [sample_dic[key][prop]
                               for key in keep_smiles]})

    smiles_folder = "/".join(train_path.split("/")[:-1])
    hyp_path = os.path.join(smiles_folder, "hyperopt_full.csv")
    write_csv(hyp_path, new_dic)


def main(base_config_path,
         hyp_config_path,
         train_folder,
         metric,
         cp_folder,
         use_hyperopt,
         rerun_hyperopt,
         max_hyp_specs,
         seed,
         **kwargs):
    """
    Load pre-set features to train a ChemProp model.
    Args:
      base_config_path (str): where your basic job config file
        is, with parameters that may or may not be changed depending
        on the given run
      hyp_config_path (str): where your basic hyperopt job config file
        is, with parameters that may or may not be changed depending
        on the given run
      train_folder (str): where you want to store your trained models
      metric (str): what metric you want to optimize in this run
      cp_folder (str): path to the chemprop folder on your computer
      use_hyperopt (bool): do a hyperparameter optimization before training
        the model
      rerun_hyperopt (bool): whether to rerun hyperparameter optimization if
        `hyp_folder` already exists and has the completion file
        `best_params.json`.
      max_hyp_specs (int): maximum number of species for hyperparameter
        optimization
      seed (int): seed for sampling data for hyperparameter optimization
    Returns:
      None
    """

    # if doing a hyperparameter optimization, run the optimization and get
    # the best parameters

    if use_hyperopt:

        hyp_folder = train_folder + "_hyp"

        modify_hyp_config(hyp_config_path=hyp_config_path,
                          metric=metric,
                          hyp_feat_path=None,
                          hyp_folder=hyp_folder,
                          features_only=None,
                          no_features=False)

        make_hyp_csvs(base_config_path=base_config_path,
                      max_specs=max_hyp_specs,
                      seed=seed)

        hyp_params = cp_hyperopt(cp_folder=cp_folder,
                                 hyp_folder=hyp_folder,
                                 rerun=rerun_hyperopt)

    else:
        hyp_params = {}

    # modify the base config with the parameters specific to this
    # run -- optimized hyperparameters, train/val/test features paths,
    # metric, etc.

    modify_config(base_config_path=base_config_path,
                  metric=metric,
                  train_feat_path=None,
                  val_feat_path=None,
                  test_feat_path=None,
                  train_folder=train_folder,
                  features_only=None,
                  hyp_params=hyp_params,
                  no_features=False)

    # train the model

    cp_train(cp_folder=cp_folder,
             train_folder=train_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_path", type=str,
                        help=("Path to the reference config file "
                              "used to train a ChemProp model. "
                              "This file will be modified with "
                              "the arguments specified below "
                              "(metric and features paths)."
                              "If they are not specified then "
                              "the config file will not be "
                              "modified."))
    parser.add_argument("--hyp_config_path", type=str, default=None,
                        help=("Same as `base_config_path`, but "
                              "for the hyperparameter optimization "
                              "stage"))
    parser.add_argument("--use_hyperopt", action='store_true',
                        help=("Do hyperparameter optimization before "
                              "training "))
    parser.add_argument("--rerun_hyperopt", action='store_true',
                        help=("Rerun hyperparameter optimization even if "
                              "it has been done already. "))

    parser.add_argument("--metric", type=str,
                        choices=CHEMPROP_METRICS,
                        help=("Metric for which to evaluate "
                              "the model performance"),
                        default=None)
    parser.add_argument("--train_folder", type=str,
                        help=("Folder in which you will store the "
                              "ChemProp model."),
                        default=None)
    parser.add_argument("--cp_folder", type=str,
                        help=("Path to ChemProp folder."))
    parser.add_argument("--max_hyp_specs", type=int,
                        help=("Maximum number of species for hyperparameter "
                              "optimization"))
    parser.add_argument("--seed", type=int,
                        help=("Seed for sampling data for hyperparameter "
                              "optimization"))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments "
                              "for this script. If given, any "
                              "arguments in the file override the "
                              "command line arguments."))

    args = parse_args(parser)
    main(**args.__dict__)
