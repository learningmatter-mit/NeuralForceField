from torch.nn import functional as F
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
#from GraphFP_qm9 import GraphDis
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import grad

from layers import * 
from module import *

class Net(nn.Module):

    """SchNet implementation with continous filter. It is designed for two types computations: 1) xyz inputs 2) graph inputs 
    
    Attributes:
        atomEmbed (torch.nn.Embedding): Convert atomic number into a embedding vector of size n_atom_basis
        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (Graphdis): graph distance module to convert xyz inputs into distance matrix 
    """
    
    def __init__(self, n_atom_basis, n_filters, n_gaussians, cutoff_soft, device, T, trainable_gauss, box_len=None, avg_flag=False):
        super(Net, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1, Fe=1, cutoff=cutoff_soft, box_len = box_len, device=device)
        self.convolutions = nn.ModuleList([InteractionBlock(n_atom_basis=n_atom_basis,
                                             n_filters=n_filters, n_gaussians=n_gaussians, 
                                             cutoff_soft =cutoff_soft, trainable_gauss=trainable_gauss) for i in range(T)])

        self.atomEmbed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.atomwise1 = Dense(in_features= n_atom_basis, out_features= int(n_atom_basis/2), activation=shifted_softplus)
        self.atomwise2 = Dense(in_features= int(n_atom_basis/2), out_features=1)
        
    def forward(self, r, xyz, a=None, N=None):
        
        # tensor inputs
        if a is None:
            
            r, e ,A = self.graph_dis(r= r, xyz=xyz)
            r = self.atomEmbed(r.type(torch.long)).squeeze()
            
            for i, conv in enumerate(self.convolutions):
                
                dr = conv(r=r, e=e, A=A)
                r = r + dr 

            r = self.atomwise1(r)
            r = self.atomwise2(r)
            r = r.sum(1)#.squeeze()
            
            return r 
        
        # graph batch inputs
        else:
            
            if N == None:
                raise ValueError("need to input N for graph partitioning within the batch")
                
            r = self.atomEmbed(r.type(torch.long)).squeeze()

            e = (xyz[a[:,0]] - xyz[a[:,1]]).pow(2).sum(1).sqrt()[:, None]

            for i, conv in enumerate(self.convolutions):
                dr = conv(r=r, e=e, a=a)
                r = r + dr

            r = self.atomwise1(r)
            r = self.atomwise2(r)

            E_batch = list(torch.split(r, N))

            for b in range(len(N)): 
                E_batch[b] = torch.sum(E_batch[b], dim=0)
            
            return torch.stack(E_batch, dim=0)#torch.Tensor(E_batch)