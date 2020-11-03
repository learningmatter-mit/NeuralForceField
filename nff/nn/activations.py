import numpy as np
import torch
import torch.nn.functional as F

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
