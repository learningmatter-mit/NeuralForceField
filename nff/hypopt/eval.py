import torch
from sklearn.metrics import (roc_auc_score,
                             auc,
                             mean_absolute_error,
                             mean_squared_error,
                             precision_recall_curve)

from nff.train import evaluate as nff_evaluate
from nff.train.chemprop import load_external_data


def pr_auc(y_true, probas_pred):
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=probas_pred)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_metric(name):
    dic = {
        "roc_auc": roc_auc_score,
        "RocAuc": roc_auc_score,
        "pr_auc": pr_auc,
        "PrAuc": pr_auc,
        "mae": mean_absolute_error,
        "mse": mean_squared_error
    }
    metric = dic[name]
    return metric


def get_eval_kwargs(model_type, param_dic):

    eval_kwargs = {}
    if "keys_to_combine" in param_dic:
        keys_to_combine = param_dic["keys_to_combine"]
        eval_kwargs.update({"keys_to_combine": keys_to_combine})

    if model_type in ["ChemProp3D", "ChemProp2D"]:
        load_dics = param_dic["chemprop"]["load_dics"]
        data, smiles_dic = load_external_data(load_dics[0])
        eval_kwargs.update({"ex_data": [data], "smiles_dics": [smiles_dic]})

    return eval_kwargs

def evaluate_model(model,
                   model_type,
                   target_name,
                   metric_name,
                   loader,
                   param_dic):

    def loss_fn(x, y): return torch.tensor(0)
    device = model.device

    eval_kwargs = get_eval_kwargs(model_type, param_dic)
    results, targets, train_loss = nff_evaluate(model,
                                                loader,
                                                loss_fn,
                                                device=device,
                                                **eval_kwargs)

    probas_pred = torch.cat(results[target_name]).reshape(-1)
    y_true = torch.cat(targets[target_name]).reshape(-1)

    metric = get_metric(metric_name)
    result = metric(y_true, probas_pred)
    if type(result) is not float:
        result = float(result)

    return result
