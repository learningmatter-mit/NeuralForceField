import numpy as np
import torch

from nff.io.ase import NeuralFF, AtomsBatch
from nff.reactive_tools.utils import *

from ase import Atoms


def powell_update(hessian_old, h, gradient_old, gradient_new):
    
    V = gradient_new - gradient_old - torch.mv(hessian_old,h)
    
    powell_hessian = hessian_old + (torch.mm(V.reshape(-1,1),h.reshape(1,-1)) 
                    + torch.mm(h.reshape(-1,1),V.reshape(1,-1)) 
                    - (torch.dot(V,h) / torch.dot(h,h) * torch.mm(h.reshape(-1,1),h.reshape(1,-1)))) / torch.dot(h,h)

    return powell_hessian.detach()


def eigvec_following(ev_atoms, step, maxstepsize, device, method,
                     hessian_old = None, gradient_old = None, h_old = None):
    
    Ndim = 3
    old_xyz = torch.Tensor(ev_atoms.get_positions()).reshape(1,-1,Ndim).to(device)
    
    if(method=='NR'or step == 0):
        hessian = torch.Tensor(neural_hessian_ase(ev_atoms)).to(device)
        
    neural_energy = neural_energy_ase(ev_atoms)
    neural_gradient = -1 * torch.Tensor(neural_force_ase(ev_atoms)).to(device)
    grad = neural_gradient.reshape(-1)
    
    
    if (method == 'Powell' and step != 0):
        hessian = powell_update(hessian_old, h_old, gradient_old, grad)
        
    #eigenvectors are stored in a transposed form
    eigenvalues, eigvecs = torch.eig(hessian, eigenvectors=True)
    
    # Ordering eigenvalues and eigenvectors in ascending order
    idx = eigenvalues[:,0].argsort()
    eigenvalues = eigenvalues[idx][:,0]
    #print(eigenvalues)
   
    eigvecs = eigvecs[:,idx]
    eigvecs_t = torch.t(eigvecs)
    #print(eigvecs_t)
    
    F = torch.mv(eigvecs_t,grad).reshape(-1,1)
    
    matrix_p = torch.Tensor([[eigenvalues[0],F[0]],[F[0],0]])

    lambda_p = torch.symeig(matrix_p, eigenvectors=False)[0][1]
    
    matrix_n = torch.zeros(Ndim*len(old_xyz[0]),Ndim*len(old_xyz[0]))
    
    for i in range(Ndim*len(old_xyz[0])-1):
        matrix_n[i][i] = eigenvalues[i+1]
        matrix_n[i][Ndim*len(old_xyz[0])-1] = F[i+1] 
        matrix_n[Ndim*len(old_xyz[0])-1][i] = F[i+1] 

    lambda_n = torch.symeig(matrix_n, eigenvectors=False)[0][0]
    
    #print(lambda_p, lambda_n) 
    lambda_n = lambda_n.new_full((Ndim*len(old_xyz[0])-1,), lambda_n.item()).to(device)
    #print(F.shape, eigvecs_t.shape, eigenvalues.shape)
    
    h_p = -1.0 * F[0]*eigvecs_t[0]/(eigenvalues[0]-lambda_p)
    h_n = -1.0 * F[1:]*eigvecs_t[1:]/((eigenvalues[1:]-lambda_n).reshape(-1,1))
    
    
    h = torch.add(h_p,torch.sum(h_n, dim=0)).reshape(-1,len(old_xyz[0]),Ndim)
    
    if(torch.max(h) <= maxstepsize):
        new_xyz = old_xyz + h
    else:
        new_xyz = old_xyz + (h / (torch.max(h) / maxstepsize))
        
    return new_xyz.detach(), grad.detach(), hessian.detach(), h.reshape(-1).detach()

def ev_run(ev_atoms, nff_dir, maxstepsize, maxstep, convergence, device, method = 'Powell'):
    rmslist = []
    maxlist = []
    
    nff = NeuralFF.from_file(nff_dir, device=device)

    for step in range(maxstep):
        if step == 0:
            xyz, grad, hessian, h = eigvec_following(ev_atoms, step, maxstepsize, device, method)
        else:
            xyz, grad, hessian, h = eigvec_following(ev_atoms,
                                                     step, maxstepsize, device, method,
                                                     hessian, grad, h)

        if step == 0:
            xyz_all = xyz
        else:
            xyz_all = torch.cat((xyz_all,xyz),dim=0)


        rmslist.append(grad.pow(2).sqrt().mean())
        maxlist.append(grad.pow(2).sqrt().max())
        print("RMS: {}, MAX: {}".format(grad.pow(2).sqrt().mean(),grad.pow(2).sqrt().max()))

        if grad.pow(2).sqrt().max() < convergence:
            print("Optimization converged!")
            break

        ev_atoms = Atoms(numbers=ev_atoms.numbers,
                        positions=xyz.reshape(-1,3).cpu().numpy(),
                        pbc=False)

        ev_atoms = AtomsBatch(ev_atoms, cutoff=5.5, nbr_torch=True, directed=True)

        ev_atoms.set_calculator(nff)

    return xyz, grad, xyz_all, rmslist, maxlist

