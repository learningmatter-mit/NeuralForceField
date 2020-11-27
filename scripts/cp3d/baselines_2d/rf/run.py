"""
Script for running hyperparameter optimization getting 
predictions from a random forest classifier.
"""

import os
import json
import argparse


import copy
from hyperopt import fmin, hp, tpe
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nff.utils import parse_args

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


def load_data(direc):
    data = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(direc, name + "_full.csv")
        with open(path, "r") as f:
            lines = f.readlines()[1:]
        smiles_list = [line.strip().split(",")[0] for line
                       in lines]
        val_list = [float(line.strip().split(",")[1]) for line
                    in lines]
        data[name] = {smiles: val for smiles, val in zip(
            smiles_list, val_list)}
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


def run_rf(space, test_or_val, seed, data):

    rf_hyperparams = {key: val for key, val in space.items()
                      if key not in MORGAN_HYPER_KEYS}
    morgan_hyperparams = {key: val for key, val in space.items()
                          if key in MORGAN_HYPER_KEYS}

    clf = RandomForestClassifier(class_weight="balanced",
                                 random_state=seed,
                                 **rf_hyperparams)

    x_train, y_train = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                                    data=data,
                                    splits=["train"],
                                    radius=morgan_hyperparams["radius"])

    x_val, y_val = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                                data=data,
                                splits=[test_or_val],
                                radius=morgan_hyperparams["radius"])

    clf.fit(x_train, y_train)
    pred_val = clf.predict(x_val)

    return pred_val, y_val, clf


def get_metrics(pred, real):

    auc_score = roc_auc_score(y_true=real, y_score=pred)
    precision, recall, thresholds = precision_recall_curve(
        y_true=real, probas_pred=pred)
    prc_score = auc(recall, precision)

    metrics = {"prc-auc": float(prc_score), "auc": float(auc_score)}

    return metrics


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
                                 data=data)
        metrics = get_metrics(pred, real)
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


def get_preds(clf, data, fp_len, radius):
    results = {}
    for name in ["train", "val", "test"]:

        x, real = make_mol_rep(fp_len=fp_len,
                               data=data,
                               splits=[name],
                               radius=radius)

        pred = clf.predict(x)
        metrics = get_metrics(pred=pred, real=real)

        results[name] = {"true": real.tolist(), "pred": pred.tolist(),
                         **metrics}

    return results


def save_preds(results, save_path):
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    print(f"All predictions saved to {save_path}")


def hyper_and_train(direc,
                    num_samples,
                    hyper_metric,
                    seed,
                    **kwargs):

    data = load_data(direc)
    objective = make_rf_objective(data, hyper_metric, seed)
    space = make_rf_space(HYPERPARAMS)

    best_params = fmin(objective,
                       space,
                       algo=tpe.suggest,
                       max_evals=num_samples)

    translate_params = translate_best_params(best_params)

    print("\n")
    print(f"Best parameters: {translate_params}")

    pred, real, clf = run_rf(translate_params,
                             test_or_val="test",
                             seed=seed,
                             data=data)
    metrics = get_metrics(pred=pred, real=real)

    print("\n")
    print(f"Test scores: {metrics}")

    results = get_preds(clf=clf,
                        data=data,
                        fp_len=translate_params["fp_len"],
                        radius=translate_params["radius"])

    save_path = os.path.join(direc, "rf_preds.json")
    save_preds(results, save_path)

    return best_params, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--direc", type=str,
                        help=("Directory with train, val and test "
                              "csv files."))

    parser.add_argument("--num_samples", type=int,
                        help=("Number of hyperparameter combinatinos "
                              "to try."))

    parser.add_argument("--hyper_metric", type=str,
                        help=("Metric to use for hyperparameter scoring."))

    parser.add_argument("--seed", type=int, default=0,
                        help=("Random seed to use."))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    hyper_and_train(**args.__dict__)
