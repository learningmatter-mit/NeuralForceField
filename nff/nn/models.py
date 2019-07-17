import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.module import GraphDis
from nff.nn.activations import shifted_softplus

class Net(nn.Module):

    """SchNet implementation with continous filter.
        It is designed for two types computations: 1) xyz inputs 2) graph inputs
        If provide bond list (bond_adj) and bond length tensor (bond_len)
        with a specified bond parameter, a harmonic bond energy 
        priors will be added  
    
    Attributes:
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis
        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        bond_energy_graph (BondEnergyModule): Description
        bond_energy_sample (BondEnergyModule): Description
        bond_par (float): Description
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (Graphdis): graph distance module to convert xyz inputs
            into distance matrix 
    """
    
    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        n_convolutions,
        cutoff,
        device='cpu',
        bond_par=50.0,
        trainable_gauss=False,
        box_size=None,
    ):
        """Constructs a SchNet model.

        Args:
            n_atom_basis (int): dimension of atomic embeddings.
            n_filters (int): dimension of filters.
            n_gaussians (int): dimension of the gaussian basis.
            n_convolutions (int): number of convolutions.
            cutoff (float): soft cutoff radius for convolution.
            bond_par (float):
            trainable_gauss (bool): if True, make the Gaussian parameter trainable.
            box_size (numpy.array): size of the box, dim = (3, )
        """

        super(Net, self).__init__()
        
        self.graph_dis = GraphDis(Fr=1,
                                  Fe=1,
                                  cutoff=cutoff,
                                  device=device,
                                  box_size=box_size)

        self.convolutions = nn.ModuleList([
            InteractionBlock(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff_soft=cutoff_soft,
                             trainable_gauss=trainable_gauss)
            for i in range(n_convolutions)
        ])

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.atomwise1 = Dense(in_features=n_atom_basis,
                               out_features=int(n_atom_basis / 2),
                               activation=shifted_softplus)

        self.atomwise2 = Dense(in_features=int(n_atom_basis / 2), out_features=1)

        # declare the bond energy module for two cases 
        self.bond_energy_graph = BondEnergyModule(batch=True)
        self.bond_energy_sample = BondEnergyModule(batch=False)
        self.bond_par = bond_par
        self.device = device

        self.to(device)
        
    def forward(self, r, xyz, a=None, N=None, bond_adj=None, bond_len=None):
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

        if a is None:
            assert len(r.shape) == 2
            assert len(xyz.shape) == 3

            r, e, A = self.graph_dis(r=r, xyz=xyz)

            r = self.atom_embed(r.type(torch.long))

            for i, conv in enumerate(self.convolutions):
                
                dr = conv(r=r, e=e, A=A)
                r = r + dr 

            r = self.atomwise1(r)
            r = self.atomwise2(r)
            r = r.sum(1)

            # compute bond energy 
            if bond_adj is not None and bond_len is not None:
                assert bond_len.shape[1] == 1

                ebond = self.bond_energy_sample(xyz=xyz,
                                               bond_adj=bond_adj,
                                               bond_len=bond_len,
                                               bond_par=self.bond_par)

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

            if N is None:
                raise ValueError("needs to input N for graph partitioning within the batch")
                
            r = self.atom_embed(r.type(torch.long)).squeeze()

            e = (xyz[a[:,0]] - xyz[a[:,1]]).pow(2).sum(1).sqrt()[:, None]

            for i, conv in enumerate(self.convolutions):
                dr = conv(r=r, e=e, a=a)
                r = r + dr

            r = self.atomwise1(r)
            r = self.atomwise2(r)

            E_batch = list(torch.split(r, N))

            # bond energy computed as a physics prior 
            if bond_adj is not None and bond_len is not None:
                ebond = self.bond_energy_graph(xyz=xyz,
                                               bond_adj=bond_adj,
                                               bond_len=bond_len,
                                               bond_par=bond_par)

                ebond_batch = list(torch.split(ebond, N))
                for b in range(len(N)): 
                    E_batch[b] = torch.sum(E_batch[b] + ebond_batch[b], dim=0)

            else:
                for b in range(len(N)): 
                    E_batch[b] = torch.sum(E_batch[b], dim=0)
                
            return torch.stack(E_batch, dim=0)
