import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.module import GraphDis

class Net(nn.Module):

    """SchNet implementation with continous filter. It is designed for two types computations: 1) xyz inputs 2) graph inputs 
    
    Attributes:
        atomEmbed (torch.nn.Embedding): Convert atomic number into a embedding vector of size n_atom_basis
        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (Graphdis): graph distance module to convert xyz inputs into distance matrix 
    """
    
    def __init__(self,
                 n_atom_basis,
                 n_filters,
                 n_gaussians,
                 cutoff_soft,
                 device,
                 T,
                 trainable_gauss,
                 box_len=None,
                 avg_flag=False):

        super(Net, self).__init__()
        
        self.graph_dis = GraphDis(
            Fr=1,
            Fe=1,
            cutoff=cutoff_soft,
            box_len=box_len,
            device=device
        )

        self.convolutions = nn.ModuleList([
            InteractionBlock(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians,
                             cutoff_soft=cutoff_soft,
                             trainable_gauss=trainable_gauss)
            for i in range(T)
        ])

        self.atomEmbed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.atomwise1 = Dense(in_features=n_atom_basis,
                               out_features=int(n_atom_basis/2),
                               activation=shifted_softplus)
        self.atomwise2 = Dense(in_features=int(n_atom_basis/2),
                               out_features=1)
        
    def forward(self, r, xyz, a=None, N=None):

        # tensor inputs
        #if a is None:
        # WUJIE: explain and refactor these functions
        if len(set(N)) == 1:
            r = r.reshape(-1, N[0])
            xyz = xyz.reshape(-1, N[0], 3)

            r, e, A = self.graph_dis(r=r, xyz=xyz)
            r = self.atomEmbed(r.type(torch.long))

            for i, conv in enumerate(self.convolutions):
                dr = conv(r=r, e=e, A=A)
                r = r + dr 

            r = self.atomwise1(r)
            r = self.atomwise2(r)
            r = r.sum(1)
            
            return r 
        
        # graph batch inputs
        else:
            assert len(r.shape) == 2
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

            # WUJIE
            for b in range(len(N)): 
                E_batch[b] = torch.sum(E_batch[b], dim=0)
            
            return torch.stack(E_batch, dim=0)

class BondNet(nn.Module):

    """SchNet implementation with continous filter.
         It is designed for two types computations: 1) xyz inputs 2) graph inputs 
    
    Attributes:
        atomEmbed (torch.nn.Embedding): Convert atomic number into a embedding vector of size n_atom_basis
        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (GraphDis): graph distance module to convert xyz inputs into distance matrix 
    """
    
    def __init__(self,
                 n_atom_basis,
                 n_filters,
                 n_gaussians,
                 cutoff_soft,
                 device,
                 T,
                 trainable_gauss,
                 box_len=None,
                 avg_flag=False,
                 bond_par=50.0):

        super(BondNet, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1,
                                  Fe=1,
                                  cutoff=cutoff_soft,
                                  box_len = box_len,
                                  device=device)

        self.convolutions = nn.ModuleList([
            InteractionBlock(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff_soft =cutoff_soft,
                             trainable_gauss=trainable_gauss)
            for i in range(T)
        ])

        self.atomEmbed = nn.Embedding(100, n_atom_basis, padding_idx=0)
        self.atomwise1 = Dense(in_features= n_atom_basis,
                               out_features= int(n_atom_basis/2),
                               activation=shifted_softplus)
        self.atomwise2 = Dense(in_features= int(n_atom_basis/2), out_features=1)

        # declare the bond energy module for two cases 
        self.bondenergy_graph = BondEnergyModule(batch=True)
        self.bondenergy_sample = BondEnergyModule(batch=False)
        self.bond_par = bond_par
        
    def forward(self, r, xyz, bond_adj=None, bond_len=None, a=None, N=None):
        """Summary
        
        Args:
            r (TYPE): Description
            xyz (TYPE): Description
            bond_adj (TYPE): Description
            a (None, optional): Description
            N (None, optional): Description
        
        Returns:
            TYPE: Description
        
        Raises:
            ValueError: Description
        """
        # tensor inputs

        bond_par = self.bond_par

        if a is None:
            assert len(r.shape) == 2
            assert len(xyz.shape) == 3

            r, e, A = self.graph_dis(r=r, xyz=xyz)

            r = self.atomEmbed(r.type(torch.long))

            for i, conv in enumerate(self.convolutions):
                
                dr = conv(r=r, e=e, A=A)
                r = r + dr 

            r = self.atomwise1(r)
            r = self.atomwise2(r)
            r = r.sum(1)

            # compute bond energy 
            if bond_adj is not None and bond_len is not None:
                ebond = self.bondenergy_sample(xyz=xyz, bond_adj=bond_adj, bond_len=bond_len, bond_par=bond_par)
                ebond = ebond.sum(1)
                return ebond + r.squeeze() 
            else:
                return r 
        
        # graph batch inputs
        else:
            assert len(r.shape) == 2 
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
            if bond_adj is not None and bond_len is not None:
                ebond = self.bondenergy_graph(xyz=xyz, bond_adj=bond_adj, bond_len=bond_len, bond_par=bond_par)
                ebond_batch = list(torch.split(ebond, N))
                for b in range(len(N)): 
                    E_batch[b] = torch.sum(E_batch[b] + ebond_batch[b], dim=0)
            else:
                for b in range(len(N)): 
                    E_batch[b] = torch.sum(E_batch[b], dim=0)
                
            return torch.stack(E_batch, dim=0)


class NetAttention(nn.Module):

    """SchNet with graph attention. It is designed for two types computations: 1) xyz inputs 2) graph inputs 
    
    Attributes:
        atomEmbed (torch.nn.Embedding): Convert atomic number into a embedding vector of size n_atom_basis
        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (Graphdis): graph distance module to convert xyz inputs into distance matrix 
    """
    
    def __init__(self,
                 n_atom_basis,
                 n_filters,
                 n_gaussians,
                 cutoff_soft,
                 device,
                 T,
                 trainable_gauss,
                 box_len=None,
                 avg_flag=False):

        super(Net_att, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1,
                                  Fe=1,
                                  cutoff=cutoff_soft,
                                  box_len = box_len,
                                  device=device)

        self.convolutions = nn.ModuleList([
            InteractionBlock(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff_soft =cutoff_soft,
                             trainable_gauss=trainable_gauss)
            for i in range(T)
        ])

        self.attentions = nn.ModuleList([graph_attention(n_atom_basis) for i in range(T)])

        self.atomEmbed = nn.Embedding(100,
                                      n_atom_basis,
                                      padding_idx=0)

        self.atomwise1 = Dense(in_features=n_atom_basis,
                               out_features= int(n_atom_basis/2),
                               activation=shifted_softplus)

        self.atomwise2 = Dense(in_features=int(n_atom_basis/2),
                               out_features=1)
        
    def forward(self, r, xyz, a=None, N=None):
        
        # tensor inputs
        if a is None:
            
            r, e, A = self.graph_dis(r= r, xyz=xyz)
            r = self.atomEmbed(r.type(torch.long)).squeeze()
            
            for i, conv in enumerate(self.convolutions):
                
                dr = conv(r=r, e=e, A=A)
                r = r + dr 
                r = self.attentions[i](r, A)

            r = self.atomwise1(r)
            r = self.atomwise2(r)
            r = r.sum(1)
            
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
            
            return torch.stack(E_batch, dim=0)
