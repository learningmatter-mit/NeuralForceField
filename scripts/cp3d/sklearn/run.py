"""
Script for running hyperparameter optimization and getting
predictions from an sklearn model.
"""

import json
import argparse
import os
from tqdm import tqdm

import copy
from hyperopt import fmin, hp, tpe
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from nff.utils import (parse_args, apply_metric, CHEMPROP_METRICS,
                       read_csv)


# load hyperparameter options for different sklearn regressors and
# classifiers

HYPER_PATH = os.path.join(os.path.abspath("."), "hyp_options.json")
with open(HYPER_PATH, "r") as f:
    HYPERPARAMS = json.load(f)

MORGAN_HYPER_KEYS = ["fp_len", "radius"]
MODEL_TYPES = list(set(list(HYPERPARAMS["classification"].keys())
                       + list(HYPERPARAMS["regression"].keys())))


def load_data(train_path,
              val_path,
              test_path):
    """
    Load data from csvs into a dictionary for the different splits.
    """
    data = {}
    paths = [train_path, val_path, test_path]
    names = ["train", "val", "test"]
    for name, path in zip(names, paths):
        data[name] = read_csv(path)

    return data


def make_mol_rep(fp_len,
                 data,
                 splits,
                 radius,
                 props):
    fps = []
    vals = []

    for split in splits:
        smiles_list = data[split]["smiles"]

        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=fp_len)

            val_list = [data[split][prop][i] for prop in props]
            # val_list = [data[split][prop][i] for prop in [props[0],
            #                                               props[0]]]
            vals.append(np.array(val_list))
            fps.append(fp)

    vals = np.stack(vals)
    # make into a 1D array if only predicting one property
    if vals.shape[-1] == 1:
        vals = vals.reshape(-1)
    fps = np.array(fps)

    return fps, vals


def get_hyperparams(model_type, classifier):
    class_or_reg = "classification" if classifier else "regression"
    hyperparams = HYPERPARAMS[class_or_reg][model_type]
    return hyperparams


def make_space(model_type, classifier):

    space = {}
    hyperparams = get_hyperparams(model_type, classifier)

    for name, sub_dic in hyperparams.items():
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


def get_splits(space,
               data,
               props):

    morgan_hyperparams = {key: val for key, val in space.items()
                          if key in MORGAN_HYPER_KEYS}

    xy_dic = {}
    for name in tqdm(["train", "val", "test"]):

        x, y = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                            data=data,
                            splits=[name],
                            radius=morgan_hyperparams["radius"],
                            props=props)

        xy_dic[name] = [x, y]

    return xy_dic


def run_sklearn(space,
                seed,
                model_type,
                classifier,
                x_train,
                y_train,
                x_test,
                y_test):

    sk_hyperparams = {key: val for key, val in space.items()
                      if key not in MORGAN_HYPER_KEYS}

    if classifier:
        if model_type == "random_forest":
            pref_fn = RandomForestClassifier(class_weight="balanced",
                                             random_state=seed,
                                             **sk_hyperparams)
        else:
            raise NotImplementedError
    else:
        if model_type == "random_forest":
            pref_fn = RandomForestRegressor(random_state=seed,
                                            **sk_hyperparams)

        else:
            raise NotImplementedError

    pref_fn.fit(x_train, y_train)
    pred_test = pref_fn.predict(x_test)

    return pred_test, y_test, pref_fn


def get_metrics(pred,
                real,
                score_metrics,
                props):

    if len(props) == 1:
        pred = pred.reshape(-1, 1)
        real = real.reshape(-1, 1)

    metric_scores = {}
    for i, prop in enumerate(props):

        metric_scores[prop] = {}

        for metric in score_metrics:

            this_pred = pred[:, i]
            this_real = real[:, i]
            score = apply_metric(metric=metric,
                                 pred=this_pred,
                                 actual=this_real)
            metric_scores[prop][metric] = float(score)

    return metric_scores


def update_saved_scores(score_path,
                        space,
                        metrics):
    if os.path.isfile(score_path):
        with open(score_path, "r") as f:
            scores = json.load(f)
    else:
        scores = []

    scores.append({**space, **metrics})

    with open(score_path, "w") as f:
        json.dump(scores, f, indent=4, sort_keys=True)


def make_objective(data,
                   metric_name,
                   seed,
                   classifier,
                   hyper_score_path,
                   model_type,
                   props):

    hyperparams = get_hyperparams(model_type, classifier)
    param_type_dic = {name: sub_dic["type"] for name, sub_dic
                      in hyperparams.items()}

    def objective(space):

        # Convert hyperparams from float to int when necessary
        for key, typ in param_type_dic.items():
            if typ == "int":
                space[key] = int(space[key])
            if isinstance(hyperparams[key]["vals"][0], bool):
                space[key] = bool(space[key])

        xy_dic = get_splits(space=space,
                            data=data,
                            props=props)

        x_val, y_val = xy_dic["val"]
        x_train, y_train = xy_dic["train"]

        pred, real, _ = run_sklearn(space=space,
                                    seed=seed,
                                    model_type=model_type,
                                    classifier=classifier,
                                    x_train=x_train,
                                    y_train=y_train,
                                    x_test=x_val,
                                    y_test=y_val)

        metrics = get_metrics(pred,
                              real,
                              [metric_name],
                              props=props)

        score = -np.mean([metrics[prop][metric_name] for prop in props])
        update_saved_scores(hyper_score_path, space, metrics)

        return score

    return objective


def translate_best_params(best_params, model_type, classifier):
    hyperparams = get_hyperparams(model_type, classifier)
    param_type_dic = {name: sub_dic["type"] for name, sub_dic
                      in hyperparams.items()}
    translate_params = copy.deepcopy(best_params)

    for key, typ in param_type_dic.items():
        if typ == "int":
            translate_params[key] = int(best_params[key])
        if typ == "categorical":
            translate_params[key] = hyperparams[key]["vals"][best_params[key]]
        if type(hyperparams[key]["vals"][0]) is bool:
            translate_params[key] = bool(best_params[key])

    return translate_params


def get_preds(pred_fn,
              score_metrics,
              xy_dic,
              props):

    results = {prop: {} for prop in props}
    for name in ["train", "val", "test"]:

        x, real = xy_dic[name]

        pred = pred_fn.predict(x)
        metrics = get_metrics(pred=pred,
                              real=real,
                              score_metrics=score_metrics,
                              props=props)

        for prop in props:
            results[prop][name] = {"true": real.tolist(),
                                   "pred": pred.tolist(),
                                   **metrics[prop]}

    return results


def save_preds(ensemble_preds,
               ensemble_scores,
               pred_save_path,
               score_save_path):

    with open(score_save_path, "w") as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    with open(pred_save_path, "w") as f:
        json.dump(ensemble_preds, f, indent=4, sort_keys=True)

    print(f"Predictions saved to {pred_save_path}")
    print(f"Scores saved to {score_save_path}")


def get_or_load_hypers(hyper_save_path,
                       rerun_hyper,
                       data,
                       hyper_metric,
                       seed,
                       classifier,
                       num_samples,
                       hyper_score_path,
                       model_type,
                       props):

    if os.path.isfile(hyper_save_path) and not rerun_hyper:
        with open(hyper_save_path, "r") as f:
            translate_params = json.load(f)
    else:

        objective = make_objective(data=data,
                                   metric_name=hyper_metric,
                                   seed=seed,
                                   classifier=classifier,
                                   hyper_score_path=hyper_score_path,
                                   model_type=model_type,
                                   props=props)

        space = make_space(model_type, classifier)

        best_params = fmin(objective,
                           space,
                           algo=tpe.suggest,
                           max_evals=num_samples,
                           rstate=np.random.RandomState(seed))

        translate_params = translate_best_params(best_params=best_params,
                                                 model_type=model_type,
                                                 classifier=classifier)
        with open(hyper_save_path, "w") as f:
            json.dump(translate_params, f, indent=4, sort_keys=True)

    print("\n")
    print(f"Best parameters: {translate_params}")

    return translate_params


def get_ensemble_preds(test_folds,
                       translate_params,
                       data,
                       classifier,
                       score_metrics,
                       model_type,
                       props):

    ensemble_preds = {}
    ensemble_scores = {}

    splits = ["train", "val", "test"]
    xy_dic = get_splits(space=translate_params,
                        data=data,
                        props=props)

    x_train, y_train = xy_dic["train"]
    x_test, y_test = xy_dic["test"]

    for seed in range(test_folds):
        pred, real, pred_fn = run_sklearn(translate_params,
                                          seed=seed,
                                          model_type=model_type,
                                          classifier=classifier,
                                          x_train=x_train,
                                          y_train=y_train,
                                          x_test=x_test,
                                          y_test=y_test)

        metrics = get_metrics(pred=pred,
                              real=real,
                              score_metrics=score_metrics,
                              props=props)

        print(f"Fold {seed} test scores: {metrics}")

        results = get_preds(pred_fn=pred_fn,
                            score_metrics=score_metrics,
                            xy_dic=xy_dic,
                            props=props)

        these_preds = {prop: {} for prop in props}
        these_scores = {prop: {} for prop in props}

        for prop in props:
            for split in splits:
                these_results = results[prop][split]
                these_scores[prop].update({split: {key: val for key, val
                                                   in these_results.items()
                                                   if key not in
                                                   ["true", "pred"]}})
                these_preds[prop].update({split: {key: val for key, val
                                                  in these_results.items()
                                                  if key in ["true", "pred"]}})

        ensemble_preds[str(seed)] = these_preds
        ensemble_scores[str(seed)] = these_scores

    avg = {prop: {split: {} for split in splits} for prop in props}

    for prop in props:

        for split in splits:

            score_dics = [sub_dic[prop][split] for sub_dic in
                          ensemble_scores.values()]

            for key in score_metrics:

                all_vals = [score_dic[key] for score_dic in score_dics]
                mean = np.mean(all_vals)
                std = np.std(all_vals)
                avg[prop][split][key] = {"mean": mean, "std": std}

    ensemble_scores["average"] = avg

    return ensemble_preds, ensemble_scores


def hyper_and_train(train_path,
                    val_path,
                    test_path,
                    pred_save_path,
                    score_save_path,
                    num_samples,
                    hyper_metric,
                    seed,
                    score_metrics,
                    hyper_save_path,
                    rerun_hyper,
                    classifier,
                    test_folds,
                    hyper_score_path,
                    model_type,
                    props,
                    **kwargs):

    data = load_data(train_path, val_path, test_path)

    translate_params = get_or_load_hypers(
        hyper_save_path=hyper_save_path,
        rerun_hyper=rerun_hyper,
        data=data,
        hyper_metric=hyper_metric,
        seed=seed,
        classifier=classifier,
        num_samples=num_samples,
        hyper_score_path=hyper_score_path,
        model_type=model_type,
        props=props)

    ensemble_preds, ensemble_scores = get_ensemble_preds(
        test_folds=test_folds,
        translate_params=translate_params,
        data=data,
        classifier=classifier,
        score_metrics=score_metrics,
        model_type=model_type,
        props=props)

    save_preds(ensemble_preds=ensemble_preds,
               ensemble_scores=ensemble_scores,
               pred_save_path=pred_save_path,
               score_save_path=score_save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str,
                        help=("Type of model you want to train"),
                        choices=MODEL_TYPES)
    parser.add_argument("--classifier", type=bool,
                        help=("Whether you're training a classifier"))
    parser.add_argument("--props", type=str, nargs="+",
                        help=("Properties for the model to predict"))
    parser.add_argument("--train_path", type=str,
                        help=("Directory to the csv with the training data"))
    parser.add_argument("--val_path", type=str,
                        help=("Directory to the csv with the validation data"))
    parser.add_argument("--test_path", type=str,
                        help=("Directory to the csv with the test data"))
    parser.add_argument("--pred_save_path", type=str,
                        help=("JSON file in which to store predictions"))
    parser.add_argument("--score_save_path", type=str,
                        help=("JSON file in which to store scores."))
    parser.add_argument("--num_samples", type=int,
                        help=("Number of hyperparameter combinatinos "
                              "to try."))
    parser.add_argument("--hyper_metric", type=str,
                        help=("Metric to use for hyperparameter scoring."))
    parser.add_argument("--hyper_save_path", type=str,
                        help=("JSON file in which to store hyperparameters"))
    parser.add_argument("--hyper_score_path", type=str,
                        help=("JSON file in which to store scores of "
                              "different hyperparameter combinations"))
    parser.add_argument("--rerun_hyper", action='store_true',
                        help=("Rerun hyperparameter optimization even "
                              "if it has already been done previously."))
    parser.add_argument("--score_metrics", type=str, nargs="+",
                        help=("Metric scores to report on test set."),
                        choices=CHEMPROP_METRICS)
    parser.add_argument("--seed", type=int,
                        help=("Random seed to use."))
    parser.add_argument("--test_folds", type=int, default=0,
                        help=("Number of different seeds to use for getting "
                              "average performance of the model on the "
                              "test set."))

    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))

    args = parse_args(parser)
    hyper_and_train(**args.__dict__)
