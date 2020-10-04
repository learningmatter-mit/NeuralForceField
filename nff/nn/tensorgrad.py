"""Summary
"""
import numpy as np

import torch
from torch.autograd import grad
from torch.autograd.gradcheck import zero_gradients


def compute_jacobian(inputs, output, device):
    """
    Compute Jacobians 
    
    Args:
        inputs (torch.Tensor): size (N_in, )
        output (torch.Tensor): size (N_in, N_out, )
        device (torch.Tensor): integer
    
    Returns:
        torch.Tensor: size (N_in, N_in, N_out)
    """
    assert inputs.requires_grad

    num_classes = output.size()[1] 
    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.to(device)
        jacobian = jacobian.to(device)

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def compute_grad(inputs, output):
    '''
    Args:
        inputs (torch.Tensor): size (N_in, )
        output (torch.Tensor): size (..., -1)
    
    Returns:
        torch.Tensor: size (N_in, )
    '''
    assert inputs.requires_grad
    
    gradspred, = grad(output,
                      inputs,
                      grad_outputs=output.data.new(output.shape).fill_(1),
                      create_graph=True,
                      retain_graph=True)
    
    return gradspred


def compute_hess(inputs, output, device):
    '''
    Compute Hessians for arbitary model
    
    Args:
        inputs (torch.Tensor): size (N_in, )
        output (torch.Tensor): size (N_out, )
        device (torch.Tensor): int
    
    Returns:
        torch.Tensor: N_in, N_in, N_out
    '''
    gradient = compute_grad(inputs, output)
    hess = compute_jacobian(inputs, gradient, device=device)
    
    return hess


def get_schnet_hessians(batch, model, device=0):
    """Get Hessians from schnet models 
    
    Args:
        batch (dict): batch of data
        model (TYPE): Description
        device (int, optional): Description
    """
    N_atom = batch['nxyz'].shape[0]
    xyz_reshape = batch["nxyz"][:, 1:].reshape(1, N_atom * 3)
    xyz_reshape.requires_grad = True
    xyz_input = xyz_reshape.reshape(N_atom, 3)
    
    r, N, xyz = model.convolve(batch, xyz_input)
    energy = model.atomwisereadout.readout["energy"](r).sum()
    hess = compute_hess(inputs=xyz_reshape, output=energy, device=device)

    return hess
