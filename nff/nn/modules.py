import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.activations import shifted_softplus


EPSILON = 1e-15


class GraphDis(torch.nn.Module):

    """Compute distance matrix on the fly 
    
    Attributes:
        box_len (numpy.array): Length of the box, dim = (3, )
        cutoff (float): cufoff 
        device (int or str): GPU id or "cpu"
        F (int): Fr + Fe
        Fe (int): edge feature length
        Fr (int): node geature length
    """
    
    def __init__(self, Fr, Fe, device, cutoff, box_len=None):
        super(GraphDis, self).__init__()

        self.Fr = Fr
        self.Fe = Fe # include distance
        self.F = Fr + Fe
        self.device = device
        self.cutoff = cutoff
        if box_len is not None:
            self.box_len = torch.Tensor(box_len).to(self.device)
        else:
            self.box_len = None
    
    def get_bond_vector_matrix(self, frame):
        """A function to compute the distance matrix 
        
        Args:
            frame (torch.FloatTensor): coordinates of (B, N, 3)
        
        Returns:
            torch.FloatTensor: distance matrix of dim (B, N, N, 1)
        """
        device = self.device
        cutoff = self.cutoff
        
        N_atom = frame.shape[1]
        frame = frame.view(-1, N_atom, 1, 3)
        dis_mat = frame.expand(-1, N_atom, N_atom, 3) \
                - frame.expand(-1, N_atom, N_atom, 3).transpose(1,2)
        
        if self.box_len is not None:

            # build minimum image convention 
            box_len = self.box_len
            mask_pos = dis_mat.ge(0.5 * box_len).type(torch.FloatTensor).to(self.device)
            mask_neg = dis_mat.lt(-0.5 * box_len).type(torch.FloatTensor).to(self.device)
            
            # modify distance 
            dis_add = mask_neg * box_len
            dis_sub = mask_pos * box_len
            dis_mat = dis_mat + dis_add - dis_sub
        
        # create cutoff mask

        # compute squared distance of dim (B, N, N)
        dis_sq = dis_mat.pow(2).sum(3)

        # mask is a byte tensor of dim (B, N, N)
        mask = (dis_sq <= cutoff ** 2) & (dis_sq != 0)

        A = mask.unsqueeze(3).type(torch.FloatTensor).to(self.device) #         

        # 1) PBC 2) # gradient of zero distance 
        dis_sq = dis_sq.unsqueeze(3)

        # to make sure the distance is not zero, otherwise there will be inf gradient 
        dis_sq = (dis_sq * A) + EPSILON
        dis_mat = dis_sq.sqrt()
        
        # compute degree of nodes 
        return(dis_mat, A.squeeze(3)) 

    def forward(self, r, xyz):

        assert len(xyz.shape) == 3
        assert xyz.shape[2] == 3

        F = self.F  # F = Fr + Fe
        Fr = self.Fr # number of node feature
        Fe = self.Fe # number of edge featue
        B = r.shape[0] # batch size
        N = r.shape[1] # number of nodes 
        device = self.device

        # Compute the bond_vector_matrix, which has shape (B, N, N, 3), and append it to the edge matrix
        e, A = self.get_bond_vector_matrix(frame=xyz)
        e = e.type(torch.FloatTensor).to(self.device)
        A = A.type(torch.FloatTensor).to(self.device)
        
        return (r, e, A)


class BondEnergyModule(nn.Module):
    
    def __init__(self, batch=True):
        super(BondEnergyModule, self).__init__()
        self.batch = batch
        
    def forward(self, xyz, bond_adj, bond_len, bond_par):
        
        if self.batch:
            e = (xyz[bond_adj[:,0]] - xyz[bond_adj[:,1]]).pow(2).sum(1).sqrt()[:, None]
            ebond = bond_par * (e - bond_len)**2
            ebond = 0.5 * scatter_add(src=ebond, index=bond_adj[:, 0], dim=0)
        else:
            e = (xyz[:, bond_adj[:,0]] - xyz[:, bond_adj[:,1]]).pow(2).sum(2).sqrt()
            ebond = 0.5 * bond_par * (e - bond_len) ** 2
        
        return ebond


class InteractionBlock(nn.Module):

    """The convolution layer with filter. To be merged with GraphConv class.
    
    Attributes:
        atom_filter (TYPE): Description
        avg_flag (Boolean): if True, perform a mean pooling 
        dense_1 (Dense()): dense layer 1 to obtain the updated atomic embedding 
        dense_2 (Dense()): dense layer 2 to obtain the updated atomic embedding
        distance_filter_1 (Dense()): dense layer 1 for filtering gaussian
            expanded distances 
        distance_filter_2 (Dense()): dense layer 1 for filtering gaussian
            expanded distances 
        smearing (GaussianSmearing()): gaussian basis expansion for distance 
            matrix of dimension B, N, N, 1
    """
    
    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        cutoff_soft,
        trainable_gauss,
        avg_flag=False
    ):

        super(InteractionBlock, self).__init__()

        self.avg_flag = avg_flag
        self.smearing = GaussianSmearing(start=0.0,
                                         stop=cutoff_soft,
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
                             out_features= n_atom_basis,
                             activation=shifted_softplus)

        self.dense_2 = Dense(in_features=n_atom_basis,
                             out_features= n_atom_basis,
                             activation=None)
        
    def forward(self, r, e, A=None, a=None):       
        if a is None: # non-batch case ...
            assert len(r.shape) == 3
            assert len(e.shape) == 4
            
            e = self.smearing(
                e.reshape(-1, r.shape[1], r.shape[1]),
                graph_batch_flag=False
            )
            W = self.distance_filter_1(e)
            W = self.distance_filter_2(e)
            r = self.atom_filter(r)
            
           # adjacency matrix used as a mask
            A = A.unsqueeze(3).expand(-1, -1, -1, r.shape[2])
            W = W #* A
            y = r[:, None, :, :].expand(-1, r.shape[1], -1, -1)
            
            # filtering 
            y = y * W * A

            # Aggregate 
            if self.avg_flag == False:
                 # sum pooling 
                y = y.sum(2)
            else:
                # mean pooling
                y = y * A.sum(2).reciprocal().expand_as(y) 
                y = y.sum(2)
            
        else:
            assert len(r.shape) == 2
            assert len(e.shape) == 2
            e = self.smearing(e, graph_batch_flag=False)
            W = self.distance_filter_1(e)
            W = self.distance_filter_2(e)
            W = W.squeeze()

            r = self.atom_filter(r)
            # Filter 
            y = r[a[:, 1]].squeeze()

            y= y * W

            # Atomwise sum 
            y = scatter_add(src=y, index=a[:, 0], dim=0)
            
        # feed into Neural networks 
        y = self.dense_1(y)
        y = self.dense_2(y)

        return y
        

class GraphAttention(nn.Module):
    def __init__(self, n_atom_basis):
        super(GraphAttention, self).__init__()
        
        self.n_atom_basis = n_atom_basis
        self.W = nn.Linear(n_atom_basis, n_atom_basis)
        self.a = nn.Linear(2 * n_atom_basis, 1)
        self.LeakyRelu = nn.RReLU()
        
    def forward(self, h, A):
        n_atom_basis = self.n_atom_basis
        B = h.shape[0]
        N_atom = h.shape[1]
        
        hi = h[:, :, None, :].expand(B, N_atom, N_atom, n_atom_basis)
        hj = h[:, None, : ,:].expand(B, N_atom, N_atom, n_atom_basis)

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
            h[:, None,  :, :].expand(B, N_atom, N_atom, n_atom_basis) * alpha
        ).sum(2)
        
        return h_prime
