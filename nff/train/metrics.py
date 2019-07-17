import numpy as np
import torch


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

    def add_batch(self, truth, prediction):
        """ Add a batch to calculate the metric on """
        self.loss += self.loss_fn(truth, prediction)
        self.n_entries += np.prod(truth.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.loss / self.n_entries

    def loss_fn(self, truth, prediction):
        """Calculates loss function for y and yp"""
        raise NotImplementedError

    def aggregate(self):
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
        super(MeanSquaredError, self).__init__(
            target=target,
            name=name,
        )

    def loss_fn(self, truth, prediction):
        diff = truth - prediction
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
        super(RootMeanSquaredError, self).__init__(
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
        super(MeanAbsoluteError, self).__init__(
            target=target,
            name=name,
        )

    def loss_fn(self, truth, prediction):
        diff = truth - prediction
        return torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()

