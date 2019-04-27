


import math
import torch
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad
#from projects.NeuralForceField.tensorgrad import *

import numpy as np

def compute_jacobian(inputs, output, device):
    """
    inputs: some variable inputs of dimesnion ( , N) that requires grad
    output: has the same dimension as inputs
    return: jacobian: dimension 
    """
    assert inputs.requires_grad

    num_classes = output.size()[1] # if the input data is batched num_classes = output.size()[1] 
    
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
    input:  some variable inputs of dimesnion ( , N) that requires grad
    output: some scalar output
    return: gradients of every input with respect to output, has the same dimension as the input
    '''
    assert inputs.requires_grad
    
    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                   create_graph=True, retain_graph=True)
    
    return gradspred

def compute_hess(inputs, output, device):
    '''
    input:  some variable inputs of dimesnion ( , N) that requires grad
    output: some scalar output
    return: Hessian of the output with respect to inputs, it is has diemension N by N
    '''
    gradient = compute_grad(inputs, output)
    hess = compute_jacobian(inputs, gradient, device=device)
    
    return hess

def Neural_hess(xyz, r, model, device, bonda=None, bondlen=None):
    
    assert len(xyz.shape) == 3
    assert len(r.shape) == 2
    assert xyz.shape[0] == r.shape[0]
    
    N_atom = xyz.shape[1]
    
    xyz_reshape = xyz.reshape(-1, N_atom * 3)
    xyz_reshape.requires_grad = True
    
    xyz_input = xyz_reshape.reshape(-1, N_atom, 3)
    U = model(r=r, xyz=xyz_input, bondlen=bondlen, bonda=bonda)

    hess = compute_hess(inputs=xyz_reshape, output=U, device=device)
    
    return hess