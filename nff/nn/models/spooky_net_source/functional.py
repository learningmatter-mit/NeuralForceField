import math
import torch

"""
IMPORTANT NOTE: Piecewise functions can be a bit tricky in pytorch:
If there are divisions by 0 in different branches of the piecewise definition,
this can lead to nan values in gradients when automatic differentiation is used
(even if the problematic branch should in theory not be evaluated at all). As a
workaround, input values for all branches must be chosen such that division by
0 does not occur. For this reason, there are some lines that may seem
unnecessary, but they are crucial for autograd to work properly.
"""

def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """ Shifted softplus activation function. """
    return torch.nn.functional.softplus(x) - math.log(2)


def cutoff_function(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Cutoff function that smoothly goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff. For x >= cutoff, f(x) = 0. This function has
    infinitely many smooth derivatives. Only positive x should be used as input.
    """
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)  # prevent nan in backprop
    return torch.where(
        x < cutoff, torch.exp(-(x_ ** 2) / ((cutoff - x_) * (cutoff + x_))), zeros
    )


def _switch_component(
    x: torch.Tensor, ones: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    """ Component of the switch function, only for internal use. """
    x_ = torch.where(x <= 0, ones, x)  # prevent nan in backprop
    return torch.where(x <= 0, zeros, torch.exp(-ones / x_))


def switch_function(x: torch.Tensor, cuton: float, cutoff: float) -> torch.Tensor:
    """
    Switch function that smoothly (and symmetrically) goes from f(x) = 1 to
    f(x) = 0 in the interval from x = cuton to x = cutoff. For x <= cuton,
    f(x) = 1 and for x >= cutoff, f(x) = 0. This switch function has infinitely
    many smooth derivatives.
    NOTE: The implementation with the "_switch_component" function is
    numerically more stable than a simplified version, it is not recommended 
    to change this!
    """
    x = (x - cuton) / (cutoff - cuton)
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1 - x, ones, zeros)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm / (fp + fm)))


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the softplus function. This is useful for initialization of
    parameters that are constrained to be positive (via softplus).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))
