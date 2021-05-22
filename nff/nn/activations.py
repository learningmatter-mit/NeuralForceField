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
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class LearnableSwish(torch.nn.Module):
    def __init__(self,
                 alpha=1,
                 beta=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
        self.beta = nn.Parameter(torch.Tensor([beta]))

    def forward(self, x):
        device = x.device
        alpha = self.alpha.to(device)
        beta = self.beta.to(device)

        output = (alpha * x) / (1 + torch.exp(-beta * x))
        return output
