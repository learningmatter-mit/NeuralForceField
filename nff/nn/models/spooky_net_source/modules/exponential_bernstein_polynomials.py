import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..functional import softplus_inverse


class ExponentialBernsteinPolynomials(nn.Module):
    """
    Radial basis functions based on exponential Bernstein polynomials given by:
    b_{v,n}(x) = (n over v) * exp(-alpha*x)**v * (1-exp(-alpha*x))**(n-v)
    (see https://en.wikipedia.org/wiki/Bernstein_polynomial)
    Here, n = num_basis_functions-1 and v takes values from 0 to n. This
    implementation operates in log space to prevent multiplication of very large
    (n over v) and very small numbers (exp(-alpha*x)**v and
    (1-exp(-alpha*x))**(n-v)) for numerical stability.
    NOTE: There is a problem for x = 0, as log(-expm1(0)) will be log(0) = -inf.
    This itself is not an issue, but the buffer v contains an entry 0 and
    0*(-inf)=nan. The correct behaviour could be recovered by replacing the nan
    with 0.0, but should not be necessary because issues are only present when
    r = 0, which will not occur with chemically meaningful inputs.

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
        """ Initializes the ExponentialBernsteinPolynomials class. """
        super(ExponentialBernsteinPolynomials, self).__init__()
        self.ini_alpha = ini_alpha
        self.exp_weighting = exp_weighting
        if no_basis_function_at_infinity:  # increase number of basis functions by one
            num_basis_functions += 1
        # compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2, num_basis_functions):
            logfactorial[i] = logfactorial[i - 1] + np.log(i)
        v = np.arange(0, num_basis_functions)
        n = (num_basis_functions - 1) - v
        logbinomial = logfactorial[-1] - logfactorial[v] - logfactorial[n]
        if no_basis_function_at_infinity:  # remove last basis function at infinity
            v = v[:-1]
            n = n[:-1]
            logbinomial = logbinomial[:-1]
        # register buffers and parameters
        self.register_buffer("logc", torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer("n", torch.tensor(n, dtype=torch.float64))
        self.register_buffer("v", torch.tensor(v, dtype=torch.float64))
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
        alphar = -F.softplus(self._alpha) * r.view(-1, 1)
        x = self.logc + self.n * alphar + self.v * torch.log(-torch.expm1(alphar))
        # x[torch.isnan(x)] = 0.0 #removes nan for r == 0, not necessary
        rbf = cutoff_values.view(-1, 1) * torch.exp(x)
        if self.exp_weighting:
            return rbf * torch.exp(alphar)
        else:
            return rbf
