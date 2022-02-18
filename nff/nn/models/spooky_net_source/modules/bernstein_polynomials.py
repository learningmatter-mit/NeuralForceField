import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..functional import softplus_inverse


class BernsteinPolynomials(nn.Module):
    """
    Radial basis functions based on Bernstein polynomials given by:
    b_{v,n}(x) = (n over v) * (x/cutoff)**v * (1-(x/cutoff))**(n-v)
    (see https://en.wikipedia.org/wiki/Bernstein_polynomial)
    Here, n = num_basis_functions-1 and v takes values from 0 to n. The basis
    functions are placed to optimally cover the range x = 0...cutoff.

    Arguments:
        num_basis_functions (int):
            Number of radial basis functions.
        cutoff (float):
            Cutoff radius.
    """

    def __init__(self, num_basis_functions: int, cutoff: float) -> None:
        """ Initializes the BernsteinPolynomials class. """
        super(BernsteinPolynomials, self).__init__()
        # compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        # register buffers and parameters
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float64))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float64))
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
        x = r.view(-1, 1) / self.cutoff
        x = torch.where(x < 1.0, x, 0.5 * torch.ones_like(x))  # prevent nans
        x = torch.log(x)
        x = self.logc + self.n * x + self.v * torch.log(-torch.expm1(x))
        # x[torch.isnan(x)] = 0.0 #removes nan for r == 0, not necessary
        rbf = cutoff_values.view(-1, 1) * torch.exp(x)
        return rbf
