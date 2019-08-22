import numpy as np

import torch
import torch.nn as nn

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.activations import shifted_softplus


EPSILON = 1e-15


class GraphDis(nn.Module):

    """Compute distance matrix on the fly 
    
    Attributes:
        Fr (int): node feature length
        Fe (int): edge feature length
        F (int): Fr + Fe
        cutoff (float): cutoff for convolution
        box_size (numpy.array): Length of the box, dim = (3, )
    """
    
    def __init__(
        self,
        Fr,
        Fe,
        cutoff,
        box_size=None
    ):
        super().__init__()

        self.Fr = Fr
        self.Fe = Fe  # include distance
        self.F = Fr + Fe
        self.cutoff = cutoff

        if box_size is not None:
            self.box_size = torch.Tensor(box_size)
        else:
            self.box_size = None
    
    def get_bond_vector_matrix(self, frame):
        """A function to compute the distance matrix 
        
        Args:
            frame (torch.FloatTensor): coordinates of (B, N, 3)
        
        Returns:
            torch.FloatTensor: distance matrix of dim (B, N, N, 1)
        """
        device = frame.device

        n_atoms = frame.shape[1]
        frame = frame.view(-1, n_atoms, 1, 3)
        dis_mat = frame.expand(-1, n_atoms, n_atoms, 3) \
            - frame.expand(-1, n_atoms, n_atoms, 3).transpose(1, 2)

        if self.box_size is not None:

            box_size = self.box_size.to(device)

            # build minimum image convention
            box_size = self.box_size
            mask_pos = dis_mat.ge(0.5 * box_size).float()
            mask_neg = dis_mat.lt(-0.5 * box_size).float()

            # modify distance
            dis_add = mask_neg * box_size
            dis_sub = mask_pos * box_size
            dis_mat = dis_mat + dis_add - dis_sub

        # create cutoff mask

        # compute squared distance of dim (B, N, N)
        dis_sq = dis_mat.pow(2).sum(3)

        # mask is a byte tensor of dim (B, N, N)
        mask = (dis_sq <= self.cutoff ** 2) & (dis_sq != 0)

        A = mask.unsqueeze(3).float()

        # 1) PBC 2) # gradient of zero distance
        dis_sq = dis_sq.unsqueeze(3)

        # to make sure the distance is not zero
        # otherwise there will be inf gradient
        dis_sq = (dis_sq * A) + EPSILON
        dis_mat = dis_sq.sqrt()

        return dis_mat, A.squeeze(3)

    def forward(self, xyz):
        e, A = self.get_bond_vector_matrix(frame=xyz)

        e = e.float()
        A = A.float()

        return e, A


class BondEnergyModule(nn.Module):
    
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch
        
    def forward(self, xyz, bond_adj, bond_len, bond_par):
        
        if self.batch:
            e = (
                xyz[bond_adj[:, 0]] - xyz[bond_adj[:, 1]]
            ).pow(2).sum(1).sqrt()[:, None]

            ebond = bond_par * (e - bond_len)**2
            ebond = 0.5 * scatter_add(src=ebond, index=bond_adj[:, 0], dim=0, dim_size=xyz.shape[0])

        else:
            e = (
                xyz[:, bond_adj[:, 0]] - xyz[:, bond_adj[:, 1]]
            ).pow(2).sum(2).sqrt()

            ebond = 0.5 * bond_par * (e - bond_len) ** 2
 
        return ebond


class InteractionBlock(nn.Module):

    """The convolution layer with filter. To be merged with GraphConv class.
    
    Attributes:
        atom_filter (Dense): Description
        dense_1 (Dense): dense layer 1 to obtain the updated atomic embedding 
        dense_2 (Dense): dense layer 2 to obtain the updated atomic embedding
        distance_filter_1 (Dense): dense layer 1 for filtering gaussian
            expanded distances
        distance_filter_2 (Dense): dense layer 1 for filtering gaussian
            expanded distances
        smearing (GaussianSmearing): gaussian basis expansion for distance 
            matrix of dimension B, N, N, 1
        mean_pooling (bool): if True, performs a mean pooling 
    """

    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        cutoff,
        trainable_gauss,
        mean_pooling=False
    ):

        super().__init__()

        self.mean_pooling = mean_pooling
        self.smearing = GaussianSmearing(start=0.0,
                                         stop=cutoff,
                                         n_gaussians=n_gaussians,
                                         trainable=trainable_gauss)

        self.distance_filter_1 = Dense(in_features=n_gaussians,
                                       out_features=n_gaussians,
                                       activation=shifted_softplus)

        self.distance_filter_2 = Dense(in_features=n_gaussians,
                                       out_features=n_filters)

        self.atom_filter = Dense(in_features=n_atom_basis,
                                 out_features=n_filters,
                                 bias=False)

        self.dense_1 = Dense(in_features=n_filters,
                             out_features=n_atom_basis,
                             activation=shifted_softplus)

        self.dense_2 = Dense(in_features=n_atom_basis,
                             out_features=n_atom_basis,
                             activation=None)
 
    def forward(self, r, e, a):
        """
        Args:
            r: feature tensor
            e: edge tensor
        """

        e = self.smearing(e, is_batch=True)
        e = self.distance_filter_1(e)
        W = self.distance_filter_2(e)
        W = W.squeeze()

        r = self.atom_filter(r)

        y = scatter_add(src=r[a[:, 0]].squeeze() * W, 
                    index=a[:, 1], 
                    dim=0, 
                    dim_size=r.shape[0])

        y += scatter_add(src=r[a[:, 1]].squeeze() * W, 
                    index=a[:, 0], 
                    dim=0, 
                    dim_size=r.shape[0])
               
        # last layers
        y = self.dense_1(y)
        y = self.dense_2(y)

        return y
        

class GraphAttention(nn.Module):
    def __init__(self, n_atom_basis):
        super().__init__()
        
        self.n_atom_basis = n_atom_basis
        self.W = nn.Linear(n_atom_basis, n_atom_basis)
        self.a = nn.Linear(2 * n_atom_basis, 1)
        self.LeakyRelu = nn.RReLU()
        
    def forward(self, h, A):
        n_atom_basis = self.n_atom_basis
        B = h.shape[0]
        N_atom = h.shape[1]
        
        hi = h[:, :, None, :].expand(B, N_atom, N_atom, n_atom_basis)
        hj = h[:, None, :, :].expand(B, N_atom, N_atom, n_atom_basis)

        hi = self.W(hi)
        hj = self.W(hj)

        # attention is directional, the attention i->j is different from j -> i
        hij = torch.cat((hi, hj), dim=3)
        hij = self.a(hij)
        
        # construct attention vector using softmax
        alpha = (torch.exp(hij) * A[:, :, :, None].expand(B, N_atom, N_atom, 1))
        alpha_sum = torch.sum(
            torch.exp(hij) * A[:, :, :, None].expand(B, N_atom, N_atom, 1),
            dim=2
        )
        alpha = alpha/alpha_sum.unsqueeze(2).expand_as(hij)

        h_prime = (
            h[:, None, :, :].expand(B, N_atom, N_atom, n_atom_basis) * alpha
        ).sum(2)
        
        return h_prime
