import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_stack import ResidualStack
from .shifted_softplus import ShiftedSoftplus
from .swish import Swish


class ResidualMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_residual: int,
        activation: str = "swish",
        bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super(ResidualMLP, self).__init__()
        self.residual = ResidualStack(
            num_features, num_residual, activation=activation, bias=bias, zero_init=True
        )
        # initialize activation function
        if activation == "ssp":
            self.activation = ShiftedSoftplus(num_features)
        elif activation == "swish":
            self.activation = Swish(num_features)
        else:
            raise ValueError(
                "Argument 'activation' may only take the "
                "values 'ssp', or 'swish' but received '" + str(activation) + "'."
            )
        self.linear = nn.Linear(num_features, num_features, bias=bias)
        self.reset_parameters(bias, zero_init)

    def reset_parameters(self, bias: bool = True, zero_init: bool = False) -> None:
        if zero_init:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.orthogonal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(self.residual(x)))
