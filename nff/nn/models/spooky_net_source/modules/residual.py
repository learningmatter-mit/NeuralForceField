import torch
import torch.nn as nn
import torch.nn.functional as F
from .shifted_softplus import ShiftedSoftplus
from .swish import Swish


class Residual(nn.Module):
    """
    Pre-activation residual block inspired by He, Kaiming, et al. "Identity
    mappings in deep residual networks.".

    Arguments:
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
        activation: str = "swish",
        bias: bool = True,
        zero_init: bool = True,
    ) -> None:
        """ Initializes the Residual class. """
        super(Residual, self).__init__()
        # initialize attributes
        if activation == "ssp":
            Activation = ShiftedSoftplus
        elif activation == "swish":
            Activation = Swish
        else:
            raise ValueError(
                "Argument 'activation' may only take the "
                "values 'ssp', or 'swish' but received '" + str(activation) + "'."
            )
        self.activation1 = Activation(num_features)
        self.linear1 = nn.Linear(num_features, num_features, bias=bias)
        self.activation2 = Activation(num_features)
        self.linear2 = nn.Linear(num_features, num_features, bias=bias)
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = True) -> None:
        """ Initialize parameters to compute an identity mapping. """
        nn.init.orthogonal_(self.linear1.weight)
        if zero_init:
            nn.init.zeros_(self.linear2.weight)
        else:
            nn.init.orthogonal_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual block to input atomic features.
        N: Number of atoms.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [N, num_features]):
                Input feature representations of atoms.

        Returns:
            y (FloatTensor [N, num_features]):
                Output feature representations of atoms.
        """
        y = self.activation1(x)
        y = self.linear1(y)
        y = self.activation2(y)
        y = self.linear2(y)
        return x + y
