import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import softplus_inverse

# backwards compatibility with older versions of torch
try:
    from torch import sinc
except:

    def sinc(x):
        x = x * math.pi
        return torch.sin(x) / x


class SincFunctions(nn.Module):
    """
    Radial basis functions based on sinc functions given by:
    g_i(x) = sinc((i+1)*x/cutoff)
    Here, i takes values from 0 to num_basis_functions-1.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        cutoff (float):
            Cutoff radius.
    """

    def __init__(self, num_basis_functions: int, cutoff: float) -> None:
        """ Initializes the SincFunctions class. """
        super(SincFunctions, self).__init__()
        self.register_buffer(
            "factor",
            torch.linspace(
                1, num_basis_functions, num_basis_functions, dtype=torch.float64
            )
            / cutoff,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def forward(self, r: torch.Tensor, cutoff_values: torch.Tensor) -> torch.Tensor:
        """
        Evaluates radial basis functions given distances and the corresponding
        values of a cutoff function (must be consistent with cutoff value
        passed at initialization).
        N: Number of input values.
        num_basis_functions: Number of radial basis functions.

        Arguments:
            r (FloatTensor [N]):
                Input distances.
            cutoff_values (FloatTensor [N]):
                Values of a cutoff function for the distances r.

        Returns:
            rbf (FloatTensor [N, num_basis_functions]):
                Values of the radial basis functions for the distances r.
        """
        x = self.factor * r.view(-1, 1)
        rbf = cutoff_values.view(-1, 1) * sinc(x)
        return rbf
