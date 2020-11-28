"""
Script for running hyperparameter optimization and getting
predictions from an sklearn model.
"""

import json
import argparse
import os

import copy
from hyperopt import fmin, hp, tpe
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from nff.utils import parse_args, apply_metric, CHEMPROP_METRICS


# load hyperparameter options for different sklearn regressors and
# classifiers

HYPER_PATH = os.path.join(os.path.abspath("."), "hyp_options.json")
with open(HYPER_PATH, "r") as f:
    HYPERPARAMS = json.load(f)

MORGAN_HYPER_KEYS = ["fp_len", "radius"]
MODEL_TYPES = list(set(list(HYPERPARAMS["classification"].keys())
                       + list(HYPERPARAMS["regression"].keys())))


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


def run_sklearn(space,
                test_or_val,
                seed,
                data,
                use_val_in_train,
                model_type,
                classifier):

    sk_hyperparams = {key: val for key, val in space.items()
                      if key not in MORGAN_HYPER_KEYS}
    morgan_hyperparams = {key: val for key, val in space.items()
                          if key in MORGAN_HYPER_KEYS}

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

    train_splits = ["train"]
    if use_val_in_train:
        train_splits.append("val")

    x_train, y_train = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                                    data=data,
                                    splits=train_splits,
                                    radius=morgan_hyperparams["radius"])

    x_val, y_val = make_mol_rep(fp_len=morgan_hyperparams["fp_len"],
                                data=data,
                                splits=[test_or_val],
                                radius=morgan_hyperparams["radius"])

    pref_fn.fit(x_train, y_train)
    pred_val = pref_fn.predict(x_val)

    return pred_val, y_val, pref_fn


def get_metrics(pred, real, score_metrics):

    metric_scores = {}
    for metric in score_metrics:
        score = apply_metric(metric=metric,
                             pred=pred,
                             actual=real)
        metric_scores[metric] = float(score)

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
                   model_type):

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

        pred, real, _ = run_sklearn(space,
                                    test_or_val="val",
                                    seed=seed,
                                    data=data,
                                    use_val_in_train=False,
                                    model_type=model_type,
                                    classifier=classifier)
        metrics = get_metrics(pred, real, [metric_name])
        score = -metrics[metric_name]
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
              data,
              fp_len,
              radius,
              score_metrics):

    results = {}
    for name in ["train", "val", "test"]:

        x, real = make_mol_rep(fp_len=fp_len,
                               data=data,
                               splits=[name],
                               radius=radius)

        pred = pred_fn.predict(x)
        metrics = get_metrics(pred=pred,
                              real=real,
                              score_metrics=score_metrics)

        results[name] = {"true": real.tolist(), "pred": pred.tolist(),
                         **metrics}

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
                       model_type):

    if os.path.isfile(hyper_save_path) and not rerun_hyper:
        with open(hyper_save_path, "r") as f:
            translate_params = json.load(f)
    else:

        objective = make_objective(data=data,
                                   metric_name=hyper_metric,
                                   seed=seed,
                                   classifier=classifier,
                                   hyper_score_path=hyper_score_path,
                                   model_type=model_type)

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
                       model_type):
    ensemble_preds = {}
    ensemble_scores = {}

    splits = ["train", "val", "test"]

    for seed in range(test_folds):
        pred, real, pred_fn = run_sklearn(translate_params,
                                          test_or_val="test",
                                          seed=seed,
                                          data=data,
                                          use_val_in_train=True,
                                          model_type=model_type,
                                          classifier=classifier)

        metrics = get_metrics(pred=pred,
                              real=real,
                              score_metrics=score_metrics)

        print(f"Fold {seed} test scores: {metrics}")

        results = get_preds(pred_fn=pred_fn,
                            data=data,
                            fp_len=translate_params["fp_len"],
                            radius=translate_params["radius"],
                            score_metrics=score_metrics)

        these_preds = {}
        these_scores = {}

        for split in splits:
            these_scores.update({split: {key: val for key, val
                                         in results[split].items()
                                         if key not in ["true", "pred"]}})
            these_preds.update({split: {key: val for key, val
                                        in results[split].items()
                                        if key in ["true", "pred"]}})

        ensemble_preds[str(seed)] = these_preds
        ensemble_scores[str(seed)] = these_scores

    avg = {split: {} for split in splits}

    for split in splits:

        score_dics = [sub_dic[split] for sub_dic in ensemble_scores.values()]

        for key in score_metrics:

            all_vals = [score_dic[key] for score_dic in score_dics]
            mean = np.mean(all_vals)
            std = np.std(all_vals)
            avg[split][key] = {"mean": mean, "std": std}

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
        model_type=model_type)

    ensemble_preds, ensemble_scores = get_ensemble_preds(
        test_folds=test_folds,
        translate_params=translate_params,
        data=data,
        classifier=classifier,
        score_metrics=score_metrics,
        model_type=model_type)

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
