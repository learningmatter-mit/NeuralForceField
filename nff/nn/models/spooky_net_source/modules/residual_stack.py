import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import Residual


class ResidualStack(nn.Module):
    """
    Stack of num_blocks pre-activation residual blocks evaluated in sequence.

    Arguments:
        num_blocks (int):
            Number of residual blocks to be stacked in sequence.
        num_features (int):
            Dimensions of feature space.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: str = "swish",
        bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        """ Initializes the ResidualStack class. """
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList(
            [
                Residual(num_features, activation, bias, zero_init)
                for i in range(num_residual)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies all residual blocks to input features in sequence.
        N: Number of inputs.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [N, num_features]):
                Input feature representations.

        Returns:
            y (FloatTensor [N, num_features]):
                Output feature representations.
        """
        for residual in self.stack:
            x = residual(x)
        return x
