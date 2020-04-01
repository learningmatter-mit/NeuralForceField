import torch
from sklearn.metrics import (roc_auc_score,
                             auc,
                             mean_absolute_error,
                             mean_squared_error,
                             precision_recall_curve)

from nff.train import evaluate as nff_evaluate


def pr_auc(y_true, probas_pred):
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_true, probas_pred=probas_pred)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_metric(name):
    dic = {
        "roc_auc": roc_auc_score,
        "pr_auc": pr_auc,
        "mae": mean_absolute_error,
        "mse": mean_squared_error
    }
    metric = dic[name]
    return metric

def evaluate_model(model, target_name, metric_name, loader):
	loss_fn = lambda x, y: torch.tensor(0)
	device = model.device
	results, targets, train_loss = nff_evaluate(model, loader, loss_fn, device=device)

	probas_pred = torch.cat(results[target_name]).reshape(-1)
	y_true = torch.cat(targets[target_name]).reshape(-1)

	metric = get_metric(metric_name)
	result = metric(y_true, probas_pred)

	return result

