import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import softplus_inverse


class GaussianFunctions(nn.Module):
    """
    Radial basis functions based on Gaussian functions given by:
    g_i(x) = exp(-width*(x-center_i)**2)
    Here, i takes values from 0 to num_basis_functions-1. The centers are chosen
    to optimally cover the range x = 0...cutoff and the width parameter is
    selected to give optimal overlap between adjacent Gaussian functions.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        cutoff (float):
            Cutoff radius.
    """

    def __init__(self, num_basis_functions: int, cutoff: float) -> None:
        """ Initializes the GaussianFunctions class. """
        super(GaussianFunctions, self).__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer(
            "center",
            torch.linspace(0, cutoff, num_basis_functions, dtype=torch.float64),
        )
        self.register_buffer(
            "width", torch.tensor(num_basis_functions / cutoff, dtype=torch.float64)
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
        rbf = cutoff_values.view(-1, 1) * torch.exp(
            -self.width * (r.view(-1, 1) - self.center) ** 2
        )
        return rbf
