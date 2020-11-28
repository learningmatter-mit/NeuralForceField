"""
Script to optimize hyperparameters for 3D models.
"""

import os
import json
import argparse
import numpy as np
from hyperopt import fmin, hp, tpe

from nff.utils import (fprint, parse_args, parse_score, convert_metric,
                       METRICS, METRIC_DIC)


def clean_up(model_path):
    """
    Clean up any files leftover from past runs with different hyperparameters.
    Args:
      model_path (str): directory of model and dataset
    Returns:
      None
    """
    cmds = ["rm */grad*.pickle",
            "rm -r checkpoints",
            "rm */train_len",
            "rm log_human_read.csv",
            "rm */log_human_read.csv",
            "rm -r best_model",
            "rm */*epoch*"]

    for cmd in cmds:
        os.system("cd {} && {}".format(model_path, cmd))


def run(job_path, model_path, metric):
    """
    Train a model with the current hyperparameters.
    Args:
      job_path (str): path to the folder with the job config file
      model_path (str): directory of model and dataset
      metric (str): metric by which to evaluate model performance
    Returns:
      best_score (float): best validation score from the model
    """

    cmd = (f"cd $NFFDIR/scripts/cp3d/train "
           f"&& python train_parallel.py {job_path}")
    os.system(cmd)
    best_score, best_epoch = parse_score(model_path, metric)

    return best_score


def update_dropout(info,
                   dropout,
                   dropout_type,
                   prop_name):
    """
    Update the config information with new dropout values.
    Args:
      info (dict): job information in the config file
      dropout (float): value of the new dropout
      dropout_type (str): type of dropout (i.e. in SchNet,
        ChemProp, readout or attention)
    Returns:
      None
    """

    if dropout_type == "schnet_dropout":
        info["model_params"]["schnet_dropout"] = dropout

    elif dropout_type == "chemprop_dropout":
        info["model_params"]["cp_dropout"] = dropout

    elif dropout_type == "readout_dropout":
        # if it's in the readout layers, find the dropout
        # layers in the readout dictionary and update them
        readout = info["model_params"]["readoutdict"]
        layer_dics = readout[prop_name]
        for layer_dic in layer_dics:
            if layer_dic["name"] == "Dropout":
                layer_dic["param"]["p"] = dropout
        info["model_params"]["readoutdict"] = {prop_name: layer_dics}

    elif dropout_type == "attention_dropout":
        info["model_params"]["boltzmann_dict"]["dropout_rate"] = dropout

    else:
        info["model_params"][dropout_type] = dropout


def update_heads(info,
                 heads):
    """
    Update the config information with the number of attention heads.
    Args:
      info (dict): job information in the config file
      heads (int): number of attention heads
    Returns:
      None
    """

    info["model_params"]["boltzmann_dict"]["num_heads"] = heads
    # Concatenate the fingerprints produced by the different heads
    info["model_params"]["boltzmann_dict"]["head_pool"] = "concatenate"

    readoutdict = info["model_params"]["readoutdict"]
    feat_dim = info["model_params"]["mol_basis"]

    for key, lst in readoutdict.items():
        for i, dic in enumerate(lst):
            if "param" in dic and "in_features" in dic.get("param", {}):
                # make sure that the input dimension to the readout is equal to
                # `heads * feat_dim`, where `feat_dim` is the feature dimension
                # produced by each head
                readoutdict[key][i]["param"]["in_features"] = feat_dim * heads
                break
    info["model_params"]["readoutdict"] = readoutdict


def update_general(info, key, val):
    """
    Update a general parameter that's in the main info dictionary.
    Args:
      info (dict): job information in the config file
      key (str): name of the parameter
      val (Union[float, str, int]): its value
    Returns:
      None
    """

    info["model_params"][key] = val


def update_info(job_path,
                vals,
                param_names,
                prop_name):
    """
    Update the config information and save it.
    Args:
      job_path (str): path to the folder with the job config file
      vals (list): new values to use
      param_names (list[str]): names of the parameters being updated
      prop_name (str): Name of property you're predicting
    Returns:
      None
    """

    with open(job_path, "r") as f:
        info = json.load(f)

    real_names = []
    real_vals = []

    for param_name, val in zip(param_names, vals):
        if param_name.startswith("log_"):
            # if anything starts with "log_" (e.g. "log_schnet_dropout"),
            # exponentiate its value to get the actual number
            real_names.append(param_name.replace("log_", ""))
            real_vals.append(np.exp(val))
        else:
            real_names.append(param_name)
            real_vals.append(val)

    # update values
    for param_type, val in zip(real_names, real_vals):
        if 'dropout' in param_type:
            update_dropout(info=info,
                           dropout=val,
                           dropout_type=param_type,
                           prop_name=prop_name)

        elif param_type == "num_heads":
            update_heads(info=info,
                         heads=val)

        elif param_type == "attention_type":
            info["model_params"]["boltzmann_dict"]["type"] = val

        else:
            if param_type not in info["model_params"]:
                msg = (f"Warning: assuming that {param_type} "
                       "is just a key in `model_params`, but "
                       "it is not currently in `model_params` in "
                       "the config file. If it should be in a "
                       "different location then you will need "
                       "to write a custom function for updating "
                       "it.")

                fprint(msg)

            update_general(info, key=param_type, val=val)

    # save
    with open(job_path, "w") as f:
        json.dump(info, f, indent=4, sort_keys=True)


def get_space(options, param_types, names):
    """
    Create a space for `hyperopt`.
    Args:
      options (list): a set of options for each parameter
      param_types (list[str]): what kind of value each parameter
        is (categorical, int or float)
      names (list): names of parameters being explored
    Returns:
      space (dict): space dictionary
    """

    space = {}

    for i, lst in enumerate(options):

        param_type = param_types[i]
        name = names[i]

        # if categorical, sample one of the options randomly
        if param_type == "categorical":
            sample = hp.choice(name, lst)

        # otherwise sample between the minimum and maximum values

        elif param_type in ["int", "float"]:

            min_val = lst[0]
            max_val = lst[1]

            if "dropout" in name:
                if min_val == 0:
                    min_val = 1e-4
                low = np.log(min_val)
                high = np.log(max_val)
                sample = hp.loguniform(name,
                                       low=low,
                                       high=high)
            elif param_type == "float":
                sample = hp.uniform(name, low=min_val, high=max_val)
            elif param_type == "int":
                sample = hp.quniform(name, low=min_val, high=max_val, q=1)

        space[name] = sample

    return space


def save_score(dic_path,
               hyperparams,
               metric,
               best_score):
    """
    Save score from a hyperparameter iteration.
    Args:
      dic_path (str): path to the JSON file with the scores
      hyperparams (dict): current hyperparameters
      metric (str): metric to optimize
      best_score (float): score from this round
    Returns:
      None
    """

    if os.path.isfile(dic_path):
        with open(dic_path, "r") as f:
            score_list = json.load(f)
    else:
        score_list = []

    score_list.append(hyperparams)
    score_list[-1].update({metric: best_score})

    with open(dic_path, "w") as f:
        json.dump(score_list, f, indent=4, sort_keys=True)


def make_objective(model_path,
                   param_names,
                   param_types,
                   job_path,
                   prop_name,
                   metric,
                   dic_path):
    """
    Make objective function that gets called by `hyperopt`.
    Args:
      model_path (str): directory of model and dataset
      param_names (list[str]): names of the parameters being updated
      param_types (list[str]): what kind of value each parameter
        is (categorical, int or float)
      job_path (str): path to the folder with the job config file
      prop_name (str): Name of property you're predicting
      num_samples (int): how many combinations of hyperparams to try
      dic_path (str): path to the JSON file with the scores
    Returns:
      objective (callable): objective function for `hyperopt`
    """

    param_type_dic = {name: typ for name, typ in zip(param_names,
                                                     param_types)}

    def objective(hyperparams):

        # clean up model folder from previous interation
        clean_up(model_path=model_path)

        # Convert hyperparams from float to int when necessary
        for key, typ in param_type_dic.items():
            if typ == "int":
                hyperparams[key] = int(hyperparams[key])

        # print hyperparameters being used
        val_str = "  " + "\n  ".join([f"{key}: {val}" for key, val
                                      in hyperparams.items()])
        fprint(f"Hyperpameters used this round:\n{val_str}")

        # update config file, run, get the score, and save
        vals = [hyperparams[key] for key in param_names]
        update_info(job_path=job_path,
                    vals=vals,
                    param_names=param_names,
                    prop_name=prop_name)

        # train the model and get the score
        best_score = run(job_path=job_path,
                         model_path=model_path,
                         metric=metric)

        # get the hyperparameter score, given that the aim is
        # to minimize whatever comes out
        metric_obj = METRIC_DIC[convert_metric(metric)]
        hyper_score = -best_score if (metric_obj ==
                                      "maximize") else best_score

        # save the score
        save_score(dic_path=dic_path,
                   hyperparams=hyperparams,
                   metric=metric,
                   best_score=best_score)

        return hyper_score

    return objective


def save_best(dic_path,
              metric,
              model_path):
    """
    Save the best parameters from the optimization.
    Args:
      dic_path (str): path to the JSON file with the scores
      metric (str): metric by which to evaluate model performance
      model_path (str): directory of model and dataset
    Returns:
      None
    """

    # load the scores
    with open(dic_path, "r") as f:
        score_list = json.load(f)

    # get the best parameters
    objective = METRIC_DIC[convert_metric(metric)]
    pref = 1 if (objective == "minimize") else (-1)
    hyper_scores = [pref * score_dic[metric] for score_dic in score_list]
    best_params = score_list[np.argmin(hyper_scores)]

    # print the best parameters
    save_path = os.path.join(model_path, "best_params.json")
    best_str = "\n  ".join([f"{key}: {val}" for key, val
                            in best_params.items()])
    fprint(f"Best parameters are {best_str}")
    fprint(f"Saving to {save_path}")

    # save them
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4, sort_keys=True)


def main(job_path,
         model_path,
         options,
         num_samples,
         metric,
         score_file,
         param_names,
         prop_name,
         param_types,
         seed,
         **kwargs):
    """
    Sample hyperparmeters and save scores for each combination.
    Args:
      job_path (str): path to the folder with the job config file
      model_path (str): directory of model and dataset
      options (list): a set of options for each parameter
      num_samples (int): how many combinations of hyperparams to try
      metric (str): metric by which to evaluate model performance
      score_file (str): name of file in which you'll store scores
      param_names (list[str]): names of the parameters being updated
      prop_name (str): Name of property you're predicting
      param_types (list[str]): what kind of value each parameter
        is (categorical, int or float)
      seed (int): random seed to use in hyperparameter optimization
    Returns:
      None
    """

    dic_path = os.path.join(model_path, score_file)

    space = get_space(options=options,
                      param_types=param_types,
                      names=param_names)

    objective = make_objective(model_path=model_path,
                               param_names=param_names,
                               param_types=param_types,
                               job_path=job_path,
                               prop_name=prop_name,
                               metric=metric,
                               dic_path=dic_path)

    # sample hyperparameters with the aim to minimize the
    # result of `objective`

    fmin(objective,
         space,
         algo=tpe.suggest,
         max_evals=num_samples,
         rstate=np.random.RandomState(seed))

    # save best results
    save_best(dic_path, metric, model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_path', type=int,
                        help=('The path with to the train config file'))
    parser.add_argument('--model_path', type=int,
                        help=("The path with the dataset where the model "
                              "will be trained"))
    parser.add_argument('--param_names', type=str, default='schnet_dropout',
                        nargs='+',
                        help=("Names of hyperparameters to optimize"))
    parser.add_argument('--param_types',
                        type=str,
                        nargs='+',
                        choices=["int", "float", "categorical"],
                        help=("Data types of hyperparameters to optimize. "
                              "Will sample uniformly between boundaries "
                              "of ints and floats, and pick random choices "
                              "for categorical parameters."))
    parser.add_argument('--options', type=str,
                        help=("Options for each parameter. Should be a list "
                              "of length 2 with the minimum and maximum "
                              "values for ints and floats. Can be a list of "
                              "any size for categorical parameters. If "
                              "using the command line, provide as a "
                              "JSON string."))
    parser.add_argument('--num_samples', type=int,
                        help=("How many hyperparameter samples to try"))
    parser.add_argument('--metric', type=str, default='prc_auc',
                        help=("Metric for judging model performance"),
                        choices=METRICS)
    parser.add_argument('--prop_name', type=str,
                        default='sars_cov_one_cl_protease_active',
                        help=("Name of property that you're predicting"))
    parser.add_argument('--score_file', type=str, default='score.json',
                        help=("Name of json file in which to save scores"))
    parser.add_argument('--seed', type=int, default=0,
                        help=("Random seed to use in hyperparameter optimization"))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    if type(args.options) is str:
        args.options = json.loads(args.options)

    main(**args.__dict__)
