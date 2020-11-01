import os
import json
import argparse
import random
import numpy as np

from nff.utils import fprint, parse_args, parse_score, METRICS


def clean_up(model_path):
    """
    Clean up any files leftover from past runs with different hyperparameters.
    Args:
      model_path (str): directory of model and dataset
    Returns:
      None
    """
    cmds = ["rm */*.pickle",
            "rm -rf check*",
            "rm */train_len",
            "rm *.csv",
            "rm */*.csv",
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


def sample_vals(options, param_types):
    """
    Do a random sample of values.
    Args:
      options (list): a set of options for each parameter
      param_types (list[str]): what kind of value each parameter
        is (categorical, int or float)
    Returns:
      vals (list): sampled values
    """

    vals = []

    for i, lst in enumerate(options):

        param_type = param_types[i]

        # if categorical, sample one of the options randomly
        if param_type == "categorical":
            val = random.choice(lst)

        # otherwise sample between the minimum and maximum values

        elif param_type in ["int", "float"]:

            min_val = lst[0]
            max_val = lst[1]

            if param_type == "float":
                val = random.uniform(float(min_val), float(max_val))
            elif param_type == "int":
                val = random.randrange(int(min_val), int(max_val) + 1)

        vals.append(val)

    return vals


def main(job_path,
         model_path,
         options,
         num_samples,
         metric,
         score_file,
         param_names,
         prop_name,
         param_types,
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
    Returns:
      None
    """

    dic_path = os.path.join(model_path, score_file)
    score_list = []

    # load old scores if they exist
    if os.path.isfile(dic_path):
        with open(dic_path, "r") as f:
            score_list = json.load(f)

    # iterate over hyperparameter combinations
    for _ in range(num_samples):

        # clean up model folder from previous interation
        clean_up(model_path=model_path)
        # sample hyperparam values
        vals = sample_vals(options, param_types)
        val_str = "  " + "\n  ".join([f"{key}: {val}" for key, val
                                      in zip(param_names, vals)])
        fprint(f"Hyperpameters used this round:\n{val_str}")

        # update config file, run, get the score, and save
        update_info(job_path=job_path,
                    vals=vals,
                    param_names=param_names,
                    prop_name=prop_name)
        best_score = run(job_path=job_path,
                         model_path=model_path,
                         metric=metric)
        score_dic = {param_type: val for param_type, val
                     in zip(param_names, vals)}
        score_dic.update({metric: best_score})
        score_list.append(score_dic)

        with open(dic_path, "w") as f:
            json.dump(score_list, f, indent=4, sort_keys=True)


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
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    if type(args.options) is str:
        args.options = json.loads(args.options)

    main(**args.__dict__)
