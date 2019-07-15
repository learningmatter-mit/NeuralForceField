import numpy as np

import torch
from torch.autograd import grad
from torch.autograd.gradcheck import zero_gradients


def compute_jacobian(inputs, output, device):
    """
    Args:
        inputs (torch.Tensor): some variable inputs of dimension (..., N).
            Requires grad.
        output (torch.Tensor): has the same dimension as inputs

    Returns:
        jacobian
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


def compute_gradient(inputs, output):
    '''
    Args:
        inputs (torch.Tensor): some variable inputs of dimension (..., N).
            Requires gradients.
        output (torch.Tensor): has the same dimension as inputs

    Returns:
        grads_pred: gradients of every input with respect to output.
            Has the same dimension as the input.
    '''
    assert inputs.requires_grad
    
    grads_pred, = grad(output, inputs,
                       grad_outputs=output.data.new(output.shape).fill_(1),
                       create_graph=True, retain_graph=True)
    
    return grads_pred


def compute_hessian(inputs, output, device):
    '''
    Args:
        inputs (torch.Tensor): some variable inputs of dimension ( , N) that requires grad
        output (torch.Tensor): scalar output

    Returns:
        hess (torch.Tensor): Hessian of the output with respect to inputs. Has dimension (..., N, N).
    '''
    gradient = compute_grad(inputs, output)
    hess = compute_jacobian(inputs, gradient, device=device)
    
    return hess


def neural_hessian(xyz, r, model, device, bond_adj=None, bond_len=None):
    
    assert len(xyz.shape) == 3
    assert len(r.shape) == 2
    assert xyz.shape[0] == r.shape[0]
    
    N_atom = xyz.shape[1]
    
    xyz_reshape = xyz.reshape(-1, N_atom * 3)
    xyz_reshape.requires_grad = True
    
    xyz_input = xyz_reshape.reshape(-1, N_atom, 3)
    U = model(r=r, xyz=xyz_input, bond_len=bond_len, bond_adj=bond_adj)

    hess = compute_hess(inputs=xyz_reshape, output=U, device=device)
    
    return hess
