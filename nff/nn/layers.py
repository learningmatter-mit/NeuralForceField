from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

from torch.nn.init import xavier_uniform_
from functools import partial
from torch.nn.init import constant_
zeros_initializer = partial(constant_, val=0.)


def shifted_softplus(x):
    """
    Shifted softplus activation function of the form:
    :math:`y = ln( e^{-x} + 1 ) - ln(2)`

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shifted softplus applied to x

    """
    return F.softplus(x) - np.log(2.0)


def gaussian_smearing(distances, offset, widths, centered=False, graph=False):
    """
    Perform gaussian smearing on interatomic distances.

    Args:
        distances (torch.Tensor): Variable holding the interatomic distances (B x N_at x N_nbh)
        offset (torch.Tensor): torch tensor of offsets
        centered (bool): If this flag is chosen, Gaussians are centered at the origin and the
                  offsets are used to provide their widths (used e.g. for angular functions).
                  Default is False.

    Returns:
        torch.Tensor: smeared distances (B x N_at x N_nbh x N_gauss)

    """
    if centered == False:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components

        if graph is not True:
            diff = distances[:, :, :, None] - offset[None, None, None, :]
        else:
            diff = distances - offset
    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        if graph is not True:
            diff = distances[:, :, :, None]
        else:
            diff = distances
    # Compute and return Gaussians

    gauss = torch.exp(coeff * torch.pow(diff, 2))

    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self, start, stop, n_gaussians, centered=False, trainable=False, graph=False):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered
        self.graph =graph

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered, graph=self.graph)


class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None,
                 weight_init=xavier_uniform_, bias_init=zeros_initializer):
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super(Dense, self).__init__(in_features, out_features, bias)

    def reset_parameters(self):
        """
        Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        y = super(Dense, self).forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y
