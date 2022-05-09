import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import softplus_inverse


class ExponentialGaussianFunctions(nn.Module):
    """
    Radial basis functions based on exponential Gaussian functions given by:
    g_i(x) = exp(-width*(exp(-alpha*x)-center_i)**2)
    Here, i takes values from 0 to num_basis_functions-1. The centers are chosen
    to optimally cover the range x = 0...infinity and the width parameter is
    selected to give optimal overlap between adjacent Gaussian functions.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        no_basis_function_at_infinity (bool):
            If True, no basis function is put at exp(-alpha*x) = 0, i.e.
            x = infinity.
        ini_alpha (float):
            Initial value for scaling parameter alpha (Default value corresponds
            to 0.5 1/Bohr converted to 1/Angstrom).
        exp_weighting (bool):
            If true, basis functions are weighted with a factor exp(-alpha*r).
    """

    def __init__(
        self,
        num_basis_functions: int,
        no_basis_function_at_infinity: bool = False,
        ini_alpha: float = 0.9448630629184640,
        exp_weighting: bool = False,
    ) -> None:
        """ Initializes the ExponentialGaussianFunctions class. """
        super(ExponentialGaussianFunctions, self).__init__()
        self.ini_alpha = ini_alpha
        self.exp_weighting = exp_weighting
        if no_basis_function_at_infinity:
            self.register_buffer(
                "center",
                torch.linspace(1, 0, num_basis_functions + 1, dtype=torch.float64)[:-1],
            )
            self.register_buffer(
                "width",
                torch.tensor(1.0 * (num_basis_functions + 1), dtype=torch.float64),
            )
        else:
            self.register_buffer(
                "center", torch.linspace(1, 0, num_basis_functions, dtype=torch.float64)
            )
            self.register_buffer(
                "width", torch.tensor(1.0 * num_basis_functions, dtype=torch.float64)
            )
        self.register_parameter(
            "_alpha", nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize exponential scaling parameter alpha. """
        nn.init.constant_(self._alpha, softplus_inverse(self.ini_alpha))

    def forward(self, r: torch.Tensor, cutoff_values: torch.Tensor) -> torch.Tensor:
        """
        Evaluates radial basis functions given distances and the corresponding
        values of a cutoff function.
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
        expalphar = torch.exp(-F.softplus(self._alpha) * r.view(-1, 1))
        rbf = cutoff_values.view(-1, 1) * torch.exp(
            -self.width * (expalphar - self.center) ** 2
        )
        if self.exp_weighting:
            return rbf * expalphar
        else:
            return rbf
