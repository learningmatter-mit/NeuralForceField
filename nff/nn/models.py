import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.module import GraphDis

class Net(nn.Module):

    """SchNet implementation with continous filter. It is designed for two types computations: 1) xyz inputs 2) graph inputs
        If provide bond list (bonda) and bond length tensor (bondlen) with a specified bond parameter, a harmonic bond energy 
        priors will be added  
    
    Attributes:
        atomEmbed (torch.nn.Embedding): Convert atomic number into a embedding vector of size n_atom_basis
        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        bondenergy_graph (BondEnergModule): Description
        bondenergy_sample (BondEnergyModule): Description
        bondpar (float): Description
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (Graphdis): graph distance module to convert xyz inputs into distance matrix 
    """
    
    def __init__(self, n_atom_basis, n_filters, n_gaussians, cutoff_soft, device, 
                    T, trainable_gauss=False, box_len=None, avg_flag=False, bondpar=50.0):
        super(BondNet, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1, Fe=1, cutoff=cutoff_soft, box_len = box_len, device=device)
        self.convolutions = nn.ModuleList([InteractionBlock(n_atom_basis=n_atom_basis,
                                             n_filters=n_filters, n_gaussians=n_gaussians, 
                                             cutoff_soft =cutoff_soft, trainable_gauss=trainable_gauss) for i in range(T)])

        self.atomEmbed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.atomwise1 = Dense(in_features= n_atom_basis, out_features= int(n_atom_basis/2), activation=shifted_softplus)
        self.atomwise2 = Dense(in_features= int(n_atom_basis/2), out_features=1)

        # declare the bond energy module for two cases 
        self.bondenergy_graph = BondEnergyModule(batch=True)
        self.bondenergy_sample = BondEnergyModule(batch=False)
        self.bondpar = bondpar
        
    def forward(self, r, xyz, a=None, N=None, bonda=None, bondlen=None):
        """Summary
        
        Args:
            r (TYPE): Description
            xyz (TYPE): Description
            bonda (TYPE): Description
            a (None, optional): Description
            N (None, optional): Description
        
        Returns:
            TYPE: Description
        
        Raises:
            ValueError: Description
        """
        # tensor inputs

        bondpar = self.bondpar

        if a is None:
            assert len(r.shape) == 2
            assert len(xyz.shape) == 3

            r, e ,A = self.graph_dis(r=r, xyz=xyz)

            r = self.atomEmbed(r.type(torch.long))#.squeeze()

            for i, conv in enumerate(self.convolutions):
                
                dr = conv(r=r, e=e, A=A)
                r = r + dr 

            r = self.atomwise1(r)
            r = self.atomwise2(r)
            r = r.sum(1)#.squeeze()

            # compute bond energy 
            if bonda is not None and bondlen is not None:
                assert bondlen.shape[1] == 1
                ebond = self.bondenergy_sample(xyz=xyz, bonda=bonda, bondlen=bondlen, bondpar=bondpar)
                ebond = ebond.sum(1)#.squeeze()
                return ebond + r.squeeze() 
            else:
                return r 
        
        # graph batch inputs
        else:
            assert len(r.shape) == 2 #1
            assert len(xyz.shape) == 2
            assert r.shape[0] == xyz.shape[0]
            assert len(a.shape) == 2

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

            # bond energy computed as a physics prior 
            if bonda is not None and bondlen is not None:
                ebond = self.bondenergy_graph(xyz=xyz, bonda=bonda, bondlen=bondlen, bondpar=bondpar)
                ebond_batch = list(torch.split(ebond, N))
                for b in range(len(N)): 
                    E_batch[b] = torch.sum(E_batch[b] + ebond_batch[b], dim=0)
            else:
                for b in range(len(N)): 
                    E_batch[b] = torch.sum(E_batch[b], dim=0)
                
            return torch.stack(E_batch, dim=0)#torch.Tensor(E_batch)