import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class shifted_softplus(torch.nn.Module):

    def __init__(self):
        super(shifted_softplus, self).__init__()

    def forward(self, input):
        return F.softplus(input) - np.log(2.0)


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x)


class LearnableSwish(torch.nn.Module):
    def __init__(self,
                 alpha=1.0,
                 beta=1.702):
        super().__init__()
        self.alpha_inv = nn.Parameter(
            torch.log(
                torch.exp(torch.Tensor([alpha])) - 1
            )
        )
        self.beta_inv = nn.Parameter(
            torch.log(
                torch.exp(torch.Tensor([beta])) - 1
            )
        )

    @property
    def alpha(self):
        return F.softplus(self.alpha_inv)

    @property
    def beta(self):
        return F.softplus(self.beta_inv)

    def forward(self, x):
        device = x.device
        alpha = self.alpha.to(device)
        beta = self.beta.to(device)
        output = alpha / beta * F.silu(beta * x)

        return output
