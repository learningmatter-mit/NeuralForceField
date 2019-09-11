import numpy as np
import torch
import torch.nn.functional as F

class shifted_softplus(torch.nn.Module):

    def __init__(self):
        super(shifted_softplus, self).__init__()

    def forward(self, input):
        return F.softplus(input) - np.log(2.0)
