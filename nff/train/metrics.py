import numpy as np
import torch
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve


class Metric:
    r"""
    Base class for all metrics.

    Metrics measure the performance during the training and evaluation.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs
            (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
    """

    def __init__(self, target, name=None):
        self.target = target
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.loss = 0.0
        self.n_entries = 0.0

    def add_batch(self, batch, results):
        """ Add a batch to calculate the metric on """

        y = batch[self.target]
        yp = results[self.target]

        self.loss += self.loss_fn(y, yp)
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.loss / self.n_entries

    @staticmethod
    def loss_fn(y, yp):
        """Calculates loss function for y and yp"""
        raise NotImplementedError


class MeanSquaredError(Metric):
    r"""
    Metric for mean square error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`,
            `MSE_[target]` will be used (Default: None)
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "MSE_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):
        diff = y - yp.view(y.shape)
        return torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()


class RootMeanSquaredError(MeanSquaredError):
    r"""
    Metric for root mean square error. For non-scalar quantities, the mean of
    all components is taken.

    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`,
            `RMSE_[target]` will be used (Default: None)
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "RMSE_" + target if name is None else name
        super().__init__(
            target, name
        )

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return np.sqrt(self.loss / self.n_entries)


class MeanAbsoluteError(Metric):
    r"""
    Metric for mean absolute error. For non-scalar quantities, the mean of all
    components is taken.

    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`,
            `MAE_[target]` will be used (Default: None)
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "MAE_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):

        # select only properties which are given
        valid_idx = torch.bitwise_not(torch.isnan(y))
        y = y[valid_idx]
        yp = yp[valid_idx]

        y = y.to(torch.float)
        diff = y - yp.view(y.shape)

        return torch.sum(torch.abs(diff).view(-1)).detach().cpu().data.numpy()


class Classifier(Metric):
    """" Metric for binary classification."""

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "Classifier_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    def add_batch(self, batch, results):
        """ Add a batch to calculate the metric on """

        y = batch[self.target]
        yp = results[self.target]

        loss, num_pred = self.loss_fn(y, yp)

        self.n_entries += num_pred
        self.loss += loss

    def non_nan(self):

        actual = torch.Tensor(self.actual)
        pred = torch.Tensor(self.pred)

        non_nan_idx = torch.bitwise_not(torch.isnan(pred))
        pred = pred[non_nan_idx].numpy().tolist()
        actual = actual[non_nan_idx].numpy().tolist()

        return pred, actual

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        if self.n_entries == 0:
            result = float('nan')
        else:
            result = self.loss / self.n_entries
        return result


class FalsePositives(Classifier):
    """
    Percentage of claimed positives that are actually wrong for a 
    binary classifier.
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "FalsePositive_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):

        actual = y.detach().cpu().numpy().round().reshape(-1)
        pred = yp.detach().cpu().numpy().round().reshape(-1)

        all_positives = [i for i, item in enumerate(pred) if item == 1]
        false_positives = [i for i in range(len(pred)) if pred[i]
                           == 1 and pred[i] != actual[i]]

        # number of predicted negatives
        num_pred = len(all_positives)
        num_pred_correct = len(false_positives)

        return num_pred_correct, num_pred


class FalseNegatives(Classifier):

    """
    Percentage of claimed negatives that are actually wrong for a 
    binary classifier.
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "FalseNegative_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):

        actual = y.detach().cpu().numpy().round().reshape(-1)
        pred = yp.detach().cpu().numpy().round().reshape(-1)

        all_negatives = [i for i, item in enumerate(pred) if item == 0]
        false_negatives = [i for i in range(len(pred)) if pred[i]
                           == 0 and pred[i] != actual[i]]
        # number of predicted negatives
        num_pred = len(all_negatives)
        num_pred_correct = len(false_negatives)

        return num_pred_correct, num_pred


class TruePositives(Classifier):

    """
    Percentage of claimed positives that are actually right for a 
    binary classifier.
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "TruePositive_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):

        actual = y.detach().cpu().numpy().round().reshape(-1)
        pred = yp.detach().cpu().numpy().round().reshape(-1)

        all_positives = [i for i, item in enumerate(pred) if item == 1]
        true_positives = [i for i in range(len(pred)) if pred[i]
                          == 1 and pred[i] == actual[i]]

        # number of predicted negatives
        num_pred = len(all_positives)
        num_pred_correct = len(true_positives)

        return num_pred_correct, num_pred


class TrueNegatives(Classifier):

    """
    Percentage of claimed negatives that are actually right for a 
    binary classifier.
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "TrueNegative_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):

        actual = y.detach().cpu().numpy().round().reshape(-1)
        pred = yp.detach().cpu().numpy().round().reshape(-1)

        all_negatives = [i for i, item in enumerate(pred) if item == 0]
        true_negatives = [i for i in range(len(pred)) if pred[i]
                          == 0 and pred[i] == actual[i]]

        # number of predicted negatives
        num_pred = len(all_negatives)
        num_pred_correct = len(true_negatives)

        return num_pred_correct, num_pred


class RocAuc(Classifier):

    """
    AUC metric (area under true-positive vs. false-positive curve).
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "RocAuc_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

        # list of actual and predicted probabilities
        self.actual = []
        self.pred = []

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""

        self.actual = []
        self.pred = []

    def loss_fn(self, y, yp):
        """The loss function here is not actually a loss function,
        but just returns actual and predicted values to add to the total.
        The AUC is calculated in the aggregate step."""

        actual = y.detach().cpu().reshape(-1).numpy().tolist()
        pred = yp.detach().cpu().reshape(-1).numpy().tolist()

        return actual, pred

    def add_batch(self, batch, results):
        """ Add a batch to calculate the metric on """

        y = batch[self.target]
        yp = results[self.target]

        actual, pred = self.loss_fn(y, yp)
        # add to actual and predicted
        self.actual += actual
        self.pred += pred

    def aggregate(self):
        """Calculate the auc score from all the data."""

        pred, actual = self.non_nan()

        try:
            auc = roc_auc_score(y_true=actual, y_score=pred)
        except ValueError:
            auc = float("nan")
        return auc


class PrAuc(Classifier):

    """
    AUC metric (area under true-positive vs. false-positive curve).
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "PrAuc_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

        # list of actual and predicted probabilities
        self.actual = []
        self.pred = []

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""

        self.actual = []
        self.pred = []

    def loss_fn(self, y, yp):
        """The loss function here is not actually a loss function,
        but just returns actual and predicted values to add to the total.
        The AUC is calculated in the aggregate step."""

        actual = y.detach().cpu().reshape(-1).numpy().tolist()
        pred = yp.detach().cpu().reshape(-1).numpy().tolist()

        return actual, pred

    def add_batch(self, batch, results):
        """ Add a batch to calculate the metric on """

        y = batch[self.target]
        yp = results[self.target]

        actual, pred = self.loss_fn(y, yp)
        # add to actual and predicted
        self.actual += actual
        self.pred += pred

    def aggregate(self):
        """Calculate the auc score from all the data."""

        pred, actual = self.non_nan()

        try:
            precision, recall, thresholds = precision_recall_curve(
                y_true=actual, probas_pred=pred)
            pr_auc = auc(recall, precision)

        except ValueError:
            pr_auc = float("nan")

        return pr_auc


class Accuracy(Classifier):

    """
    Overall accuracy of classifier.
    """

    def __init__(
        self,
        target,
        name=None,
    ):
        name = "Accuracy_" + target if name is None else name
        super().__init__(
            target=target,
            name=name,
        )

    @staticmethod
    def loss_fn(y, yp):

        actual = y.detach().cpu().numpy().round().reshape(-1)
        pred = yp.detach().cpu().numpy().round().reshape(-1)

        # number of predicted negatives
        num_pred = len(actual)
        correct = [i for i in range(num_pred) if actual[i] == pred[i]]
        num_pred_correct = len(correct)

        return num_pred_correct, num_pred
