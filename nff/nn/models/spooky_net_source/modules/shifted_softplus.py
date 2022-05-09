import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedSoftplus(nn.Module):
    """
    Shifted softplus activation function with learnable feature-wise parameters:
    f(x) = alpha/beta * (softplus(beta*x) - log(2))
    softplus(x) = log(exp(x) + 1)
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        num_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the softplus function.
        initial_beta (float):
            Initial "temperature" beta of the softplus function.
    """

    def __init__(
        self, num_features: int, initial_alpha: float = 1.0, initial_beta: float = 1.0
    ) -> None:
        """ Initializes the ShiftedSoftplus class. """
        super(ShiftedSoftplus, self).__init__()
        self._log2 = math.log(2)
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(num_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters alpha and beta. """
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, num_features]):
                Input features.

        Returns:
            y (FloatTensor [:, num_features]):
                Activated features.
        """
        return self.alpha * torch.where(
            self.beta != 0,
            (F.softplus(self.beta * x) - self._log2) / self.beta,
            0.5 * x,
        )
