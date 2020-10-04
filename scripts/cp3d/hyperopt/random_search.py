import os
import json
import argparse
import random

from nff.utils import fprint, parse_args, parse_score, METRICS


def clean_up(model_path):
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

    cmd = f"cd $NFFDIR/scripts/cp3d/train && bash python $NFFDIR/scripts/cp3d/train/train_parallel.py {job_path}"
    os.system(cmd)
    best_score, best_epoch = parse_score(job_path, model_path, metric)

    return best_score


def update_dropout(info,
                   dropout,
                   dropout_type,
                   prop_name):

    if dropout_type == "schnet_dropout":
        info["model_params"]["schnet_dropout"] = dropout

    elif dropout_type == "chemprop_dropout":
        info["model_params"]["cp_dropout"] = dropout

    elif dropout_type == "readout_dropout":
        readout = info["model_params"]["readoutdict"]
        layer_dics = readout[prop_name]
        for layer_dic in layer_dics:
            if layer_dic["name"] == "Dropout":
                layer_dic["param"]["p"] = dropout
        info["model_params"]["readoutdict"] = {prop_name: layer_dics}

    else:
        info["model_params"][dropout_type] = dropout


def update_heads(info,
                 heads):

    info["model_params"]["boltzmann_dict"]["num_heads"] = heads
    info["model_params"]["boltzmann_dict"]["head_pool"] = "concatenate"

    readoutdict = info["model_params"]["readoutdict"]
    input_layers = info["model_params"]["input_layers"]
    feat_dim = input_layers[0]["param"]["out_features"]

    for key, lst in readoutdict.items():
        for i, dic in enumerate(lst):
            if "param" in dic and "in_features" in dic.get("param", {}):
                readoutdict[key][i]["param"]["in_features"] = feat_dim * heads
                break
    info["model_params"]["readoutdict"] = readoutdict


def update_general(info, key, val):

    info["model_params"][key] = val


def update_info(job_path,
                vals,
                param_names,
                prop_name):

    with open(job_path, "r") as f:
        info = json.load(f)

    for param_type, val in zip(param_names, vals):
        if 'dropout' in param_type:
            update_dropout(info=info,
                           dropout=val,
                           dropout_type=param_type,
                           prop_name=prop_name)

        elif param_type == "num_heads":
            update_heads(info=info,
                         heads=val)

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

    with open(job_path, "w") as f:
        json.dump(info, f, indent=4, sort_keys=True)


def sample_vals(options, param_types):

    vals = []

    for i, lst in enumerate(options):

        param_type = param_types[i]

        if param_type == "categorical":
            val = random.choice(lst)

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

    dic_path = os.path.join(model_path, score_file)
    score_list = []

    if os.path.isfile(dic_path):
        with open(dic_path, "r") as f:
            score_list = json.load(f)

    for _ in range(num_samples):

        clean_up(model_path=model_path)

        vals = sample_vals(options, param_types)
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
