import os
import json
import argparse
import random


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


def parse_prc_auc(job_path, model_path):

    log_path = os.path.join(model_path, "log_human_read.csv")
    with open(log_path, "r") as f:
        lines = f.readlines()

    aucs = []
    for line in reversed(lines):
        try:
            aucs.append(float(line.split("|")[-2]))
        except:
            continue

    best_auc = max(aucs)
    return best_auc


def run(job_path, model_path, metric):

    metric_func_dic = {"prc_auc": parse_prc_auc}
    metric_func = metric_func_dic[metric]

    cmd = "cd $NFFDIR/scripts/cp3d/train && bash train_parallel.sh"
    os.system(cmd)
    best_score = metric_func(job_path=job_path,
                             model_path=model_path)

    return best_score


def update_dropout(info,
                   dropout,
                   dropout_type,
                   prop_name):

    if dropout_type == "schnet_dropout":
        info["schnet_dropout"] = dropout
        info["details"]["schnet_dropout"] = dropout

    elif dropout_type == "chemprop_dropout":
        info["cp_dropout"] = dropout
        info["details"]["cp_dropout"] = dropout

    elif dropout_type == "readout_dropout":
        readout = info["details"]["readoutdict"]
        layer_dics = readout[prop_name]
        for layer_dic in layer_dics:
            if layer_dic["name"] == "Dropout":
                layer_dic["param"]["p"] = dropout
        info["details"]["readoutdict"] = {prop_name: layer_dics}


def update_heads(info,
                 heads):

    info["details"]["boltzmann_dict"]["num_heads"] = heads
    info["details"]["boltzmann_dict"]["head_pool"] = "concatenate"

    readoutdict = info["details"]["readoutdict"]
    input_layers = info["details"]["input_layers"]
    feat_dim = input_layers[0]["param"]["out_features"]

    for key, lst in readoutdict.items():
        for i, dic in enumerate(lst):
            if "param" in dic and "in_features" in dic.get("param", {}):
                readoutdict[key][i]["param"]["in_features"] = feat_dim * heads
                break
    info["details"]["readoutdict"] = readoutdict


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

    with open(job_path, "w") as f:
        json.dump(info, f, indent=4, sort_keys=True)


def sample_vals(min_vals, max_vals, data_types):

    nums = []

    for i, min_val in enumerate(min_vals):
        max_val = max_vals[i]
        data_type = eval(data_types[i])
        if data_type == float:
            num = random.uniform(min_val, max_val)
        elif data_type == int:
            num = random.randrange(min_val, max_val)
        nums.append(num)
    return nums


def main(job_path,
         model_path,
         min_vals,
         max_vals,
         num_samples,
         metric,
         score_file,
         param_names,
         prop_name,
         data_types,
         **kwargs):

    dic_path = os.path.join(model_path, score_file)
    score_list = []

    for _ in range(num_samples):

        clean_up(model_path=model_path)

        vals = sample_vals(min_vals, max_vals, data_types)
        update_info(job_path=job_path,
                    vals=vals,
                    param_names=param_names,
                    prop_name=prop_name)
        best_score = run(job_path=job_path,
                         model_path=model_path,
                         metric=metric)
        score_dic = {param_type: val for param_type, val
                     in zip(param_names, vals)}
        score_dic.update({"score": best_score})
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
    parser.add_argument('--data_types', type=str,
                        nargs='+',
                        help=("Data types of hyperparameters to optimize"))
    parser.add_argument('--min_vals', type=int,
                        help=("Minimum values of parameters in grid search"),
                        nargs='+')
    parser.add_argument('--max_vals', type=int,
                        help=("Maximum values of parameters in grid search"),
                        nargs='+')
    parser.add_argument('--num_samples', type=int,
                        help=("How many hyperparameter samples to try"))
    parser.add_argument('--metric', type=str, default='prc_auc',
                        help=("Metric for judging model performance"))
    parser.add_argument('--prop_name', type=str,
                        default='sars_cov_one_cl_protease_active',
                        help=("Name of property that you're predicting"))
    parser.add_argument('--score_file', type=str, default='score.json',
                        help=("Name of json file in which to save scores"))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            config_args = json.load(f)
        for key, val in config_args.items():
            if hasattr(args, key):
                setattr(args, key, val)

    main(**args.__dict__)
