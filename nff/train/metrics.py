import numpy as np
import torch
# import pdb

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
        # pdb.set_trace()
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
        delta = pred - actual

        # if pred - actual > 0, then you predicted positive
        # but it was negative
        false_positives = list(filter(lambda x: x > 0, delta))
        # number of predicted positive is the sum of the total
        # predictions
        num_pred = np.sum(pred)
        num_pred_false = np.sum(false_positives)

        return num_pred_false, num_pred


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
        delta = pred - actual

        # if pred - actual < 0, then you predicted negative
        # but it was positive
        false_negatives = list(filter(lambda x: x < 0, delta))
        # number of predicted positive is the sum of the total
        # predictions, so len(pred) minus this is
        # the number of negative predictions
        num_pred = len(pred) - np.sum(pred)
        num_pred_false = -np.sum(false_negatives)


        return num_pred_false, num_pred


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
        delta = pred - actual

        # if the prediction is 1 and delta = 0 then it's
        # a correct positive
        true_positives = [i for i, diff in enumerate(
            delta) if diff == 0 and pred[i] == 1]
        # number of predicted positives
        num_pred = np.sum(pred)
        num_pred_correct = np.sum(true_positives)

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
        delta = pred - actual

        # if the prediction is 0 and delta = 0 then it's
        # a correct negative
        true_negatives = [i for i, diff in enumerate(
            delta) if diff == 0 and pred[i] == 0]
        # number of predicted negatives
        num_pred = len(pred) - np.sum(pred)
        num_pred_correct = len(true_negatives)

        return num_pred_correct, num_pred

