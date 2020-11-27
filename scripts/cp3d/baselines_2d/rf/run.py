"""
Script for running hyperparameter optimization getting
predictions from a random forest classifier.
"""

import json
import argparse
import os

import copy
from hyperopt import fmin, hp, tpe
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nff.utils import parse_args, apply_metric, CHEMPROP_METRICS

MORGAN_HYPER_KEYS = ["fp_len", "radius"]

HYPERPARAMS = {"n_estimators": {"vals": [20, 300], "type": "int"},
               "criterion": {"vals": ["gini", "entropy"],
                             "type": "categorical"},
               "min_samples_split": {"vals": [2, 10], "type": "int"},
               "min_samples_leaf": {"vals": [1, 5],
                                    "type": "int"},
               "min_weight_fraction_leaf": {"vals": [0.0, 0.5],
                                            "type": "float"},
               "max_features": {"vals": ["auto", "sqrt", "log2"],
                                "type": "categorical"},
               "min_impurity_decrease": {"vals": [0.0, 0.5],
                                         "type": "float"},
               "max_samples": {"vals": [1e-5, 1 - 1e-5],
                               "type": "float"},
               "fp_len": {"vals": [64, 128, 256, 1024, 2048],
                          "type": "categorical"},
               "radius": {"vals": [1, 4], "type": "int"}}


def load_data(train_path, val_path, test_path):
    data = {}
    paths = [train_path, val_path, test_path]
    names = ["train", "val", "test"]
    for name, path in zip(names, paths):
        with open(path, "r") as f:
            lines = f.readlines()[1:]
            smiles_list = [line.strip().split(",")[0] for line in lines]
            val_list = [float(line.strip().split(",")[1]) for line in lines]
            data[name] = {smiles: val for smiles,
                          val in zip(smiles_list, val_list)}
    return data


def make_mol_rep(fp_len, data, splits, radius):

    fps = []
    vals = []
    for split in splits:
        for smiles, val in data[split].items():
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=fp_len)

            vals.append(val)
            fps.append(fp)

    vals = np.array(vals)
    fps = np.array(fps)

    return fps, vals


def make_rf_space(HYPERPARAMS):
    space = {}
    for name, sub_dic in HYPERPARAMS.items():
        val_type = sub_dic["type"]
        vals = sub_dic["vals"]
        if val_type == "categorical":
            sample = hp.choice(name, vals)
        elif val_type == "float":
            sample = hp.uniform(name, low=float(min(vals)),
                                high=float(max(vals)))
        elif val_type == "int":
            sample = hp.quniform(name, low=min(vals),
                                 high=max(vals), q=1)
        space[name] = sample
    return space


def run_rf(space,
           test_or_val,
           seed,
           data,
           use_val_in_train):

    rf_hyperparams = {key: val for key, val in space.items()
                      if key not in MORGAN_HYPER_KEYS}
    morgan_hyperparams = {key: val for key, val in space.items()
                          if key in MORGAN_HYPER_KEYS}

    clf = RandomForestClassifier(class_weight="balanced",
                                 random_state=seed,
                                 **rf_hyperparams)

    train_splits = ["train"]
    if use_val_in_train:
        train_splits.append(["val"])

    x_train, y_train = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                                    data=data,
                                    splits=train_splits,
                                    radius=morgan_hyperparams["radius"])

    x_val, y_val = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                                data=data,
                                splits=[test_or_val],
                                radius=morgan_hyperparams["radius"])

    clf.fit(x_train, y_train)
    pred_val = clf.predict(x_val)

    return pred_val, y_val, clf


def get_metrics(pred, real, score_metrics):

    metric_scores = {}
    for metric in score_metrics:
        score = apply_metric(metric=metric,
                             pred=pred,
                             actual=real)
        metric_scores[metric] = float(score)

    return metric_scores


def make_rf_objective(data, metric_name, seed):
    param_type_dic = {name: sub_dic["type"] for name, sub_dic
                      in HYPERPARAMS.items()}

    def objective(space):
        # Convert HYPERPARAMS from float to int when necessary
        for key, typ in param_type_dic.items():
            if typ == "int":
                space[key] = int(space[key])
            if type(HYPERPARAMS[key]["vals"][0]) is bool:
                space[key] = bool(space[key])

        pred, real, clf = run_rf(space,
                                 test_or_val="val",
                                 seed=seed,
                                 data=data,
                                 use_val_in_train=False)
        metrics = get_metrics(pred, real, [metric_name])
        score = -metrics[metric_name]

        return score

    return objective


def translate_best_params(best_params):
    param_type_dic = {name: sub_dic["type"] for name, sub_dic
                      in HYPERPARAMS.items()}
    translate_params = copy.deepcopy(best_params)

    for key, typ in param_type_dic.items():
        if typ == "int":
            translate_params[key] = int(best_params[key])
        if typ == "categorical":
            translate_params[key] = HYPERPARAMS[key]["vals"][best_params[key]]
        if type(HYPERPARAMS[key]["vals"][0]) is bool:
            translate_params[key] = bool(best_params[key])

    return translate_params


def get_preds(clf, data, fp_len, radius, score_metrics):
    results = {}
    for name in ["train", "val", "test"]:

        x, real = make_mol_rep(fp_len=fp_len,
                               data=data,
                               splits=[name],
                               radius=radius)

        pred = clf.predict(x)
        metrics = get_metrics(pred=pred,
                              real=real,
                              score_metrics=score_metrics)

        results[name] = {"true": real.tolist(), "pred": pred.tolist(),
                         **metrics}

    return results


def save_preds(results, save_path):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    print(f"All predictions saved to {save_path}")


def hyper_and_train(train_path,
                    val_path,
                    test_path,
                    save_path,
                    num_samples,
                    hyper_metric,
                    seed,
                    score_metrics,
                    hyper_save_path,
                    rerun_hyper,
                    **kwargs):

    data = load_data(train_path, val_path, test_path)

    if os.path.isfile(hyper_save_path) and not rerun_hyper:
        with open(hyper_save_path, "r") as f:
            translate_params = json.load(f)
    else:
        objective = make_rf_objective(data, hyper_metric, seed)
        space = make_rf_space(HYPERPARAMS)

        best_params = fmin(objective,
                           space,
                           algo=tpe.suggest,
                           max_evals=num_samples)

        translate_params = translate_best_params(best_params)
        with open(hyper_save_path, "w") as f:
            json.dump(translate_params, f, indent=4, sort_keys=True)

    print("\n")
    print(f"Best parameters: {translate_params}")

    pred, real, clf = run_rf(translate_params,
                             test_or_val="test",
                             seed=seed,
                             data=data,
                             use_val_in_train=True)
    metrics = get_metrics(pred=pred, real=real,
                          score_metrics=score_metrics)

    print("\n")
    print(f"Test scores: {metrics}")

    results = get_preds(clf=clf,
                        data=data,
                        fp_len=translate_params["fp_len"],
                        radius=translate_params["radius"],
                        score_metrics=score_metrics)

    save_preds(results, save_path)

    return best_params, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str,
                        help=("Directory to the csv with the training data"))
    parser.add_argument("--val_path", type=str,
                        help=("Directory to the csv with the validation data"))
    parser.add_argument("--test_path", type=str,
                        help=("Directory to the csv with the test data"))
    parser.add_argument("--save_path", type=str,
                        help=("JSON file in which to store predictions"))
    parser.add_argument("--num_samples", type=int,
                        help=("Number of hyperparameter combinatinos "
                              "to try."))
    parser.add_argument("--hyper_metric", type=str,
                        help=("Metric to use for hyperparameter scoring."))
    parser.add_argument("--hyper_save_path", type=str,
                        help=("JSON file in which to store hyperparameters"))
    parser.add_argument("--rerun_hyper", action='store_true',
                        help=("Rerun hyperparameter optimization even "
                              "if it has already been done previously."))
    parser.add_argument("--score_metrics", type=str, nargs="+",
                        help=("Metric scores to report on test set."),
                        choices=CHEMPROP_METRICS)
    parser.add_argument("--seed", type=int, default=0,
                        help=("Random seed to use."))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    hyper_and_train(**args.__dict__)
