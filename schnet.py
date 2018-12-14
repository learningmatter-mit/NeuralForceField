from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
from GraphFP_qm9 import GraphDis
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad

def compute_grad(inputs, output):
    """compute gradient of the scalar output with respect to inputs 
    
    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 
    
    Returns:
        torch.Tensor: gradients with respect to each input component 
    """
    assert inputs.requires_grad
    
    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                   create_graph=True, retain_graph=True)
    
    return gradspred

# Gaussian Smearing 
def gaussian_smearing(distances, offset, widths, centered=False):
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
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances[:, :, :, None]
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

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
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

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)

class Dense(nn.Module):
    """ 
    Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(Dense, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation

    def forward(self, inputs):
        if self.activation is not None:
            return self.activation(self.linear(inputs))
        else: 
            return self.linear(inputs)

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


class InteractionBlock(nn.Module):
    """ 
    Interaction Block with a distance filter based on Gaussian smearing 

    Args:
        n_atom_basis (int): number of atom features
        n_filters (int): filter dimensions
        n_gaussians (int): number of guassian basis
        
    """

    def __init__(self, n_atom_basis, n_filters, n_gaussians):
        super(InteractionBlock, self).__init__()
        self.smearing = GaussianSmearing(start=0.0, stop=5.0, n_gaussians=n_gaussians, trainable=True)
        self.DistanceFilter1 = Dense(in_features= n_gaussians, out_features=n_gaussians, activation=shifted_softplus)
        self.DistanceFilter2 = Dense(in_features= n_gaussians, out_features=n_filters)
        self.Dense1 = Dense(in_features=n_filters, out_features= n_atom_basis, activation=shifted_softplus)
        self.Dense2 = Dense(in_features=n_filters, out_features= n_atom_basis, activation=None)
        
    def forward(self, r, e):
        e = self.smearing(e.squeeze())
        W = self.DistanceFilter1(e)
        W = self.DistanceFilter2(e)
        
        #r = r[:, None, :, :].expand(-1, r.shape[1], -1, -1) # expand to filtered 
        #print(W.shape)
        y = r[:, None, :, :].expand(-1, r.shape[1], -1, -1) * W
        
        y = self.Dense1(y)
        y = self.Dense2(y)
        
        y = y.sum(2) # sum pooling 
        
        return y

class Net(nn.Module):
    """ 
    Module to compute energy 

    Args:
        n_atom_basis (int): number of atom features
        n_filters (int): filter dimensions
        n_gaussians (int): number of guassian basis
        device (int): which gpu to use
        
    """

    def __init__(self, n_atom_basis, n_filters, n_gaussians, cutoff, device):
        super(Net, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1, Fe=1, cutoff=cutoff, device=device)
        self.interaction1 = InteractionBlock(n_atom_basis=n_atom_basis, n_filters=n_filters, n_gaussians=n_gaussians)
        self.interaction2 = InteractionBlock(n_atom_basis=n_atom_basis, n_filters=n_filters, n_gaussians=n_gaussians)
        self.interaction3 = InteractionBlock(n_atom_basis=n_atom_basis, n_filters=n_filters, n_gaussians=n_gaussians)
        self.atomEmbed = Dense(in_features= 1, out_features=n_atom_basis, bias=False)
        self.atomwise1 = Dense(in_features= n_atom_basis, out_features= 10)
        self.atomwise2 = Dense(in_features= 10, out_features=1)
        
        self.ssp = shifted_softplus
        
    def forward(self, r, xyz):
        
        r, e ,A = self.graph_dis(r= r, xyz=xyz)
        r = self.atomEmbed(r)
        
        r = r + self.interaction1(r=r, e=e)
        #r = r + self.interaction2(r=r, e=e)
        #r = r + self.interaction3(r=r, e=e)

        r = self.atomwise1(r)
        r = self.ssp(r)
        r = self.atomwise2(r)
        
        r = r.sum(1)#.squeeze()
        return r