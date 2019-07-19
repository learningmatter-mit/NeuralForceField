import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, constant_


zeros_initializer = partial(constant_, val=0.)


def gaussian_smearing(distances, offset, widths, centered=False, is_batch=False):

    if centered == False:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)

        # Use advanced indexing to compute the individual components
        if not is_batch:
            diff = distances[:, :, :, None] - offset[None, None, None, :]
        else:
            diff = distances - offset

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)

        # If centered Gaussians are requested, don't substract anything
        if not is_batch:
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

    def __init__(self, start, stop, n_gaussians, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances, is_batch):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        result = gaussian_smearing(distances,
                                   self.offsets,
                                   self.width,
                                   centered=self.centered,
                                   is_batch=is_batch)

        return result


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

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init
        self.activation = activation

        super().__init__(in_features, out_features, bias)

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
        y = super().forward(inputs)
        if self.activation:
            y = self.activation(y)

        return y
