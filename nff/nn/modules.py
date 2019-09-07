import numpy as np

import torch
import torch.nn as nn

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.activations import shifted_softplus
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict 
from nff.nn.graphconv import MessagePassingModule, EdgeUpdateModule, \
                            GeometricOperations, TopologyOperations
from nff.utils.tools import construct_Sequential, construct_ModuleDict

import unittest

EPSILON = 1e-15

class SchNetEdgeUpdate(EdgeUpdateModule):
    """
    Arxiv.1806.03146
    
    Attributes:
        mlp (TYPE): Update function 
    """
    def __init__(self, n_atom_basis):
        super(SchNetEdgeUpdate, self).__init__()

        self.mlp = Sequential(
                              Linear(2 * n_atom_basis, n_atom_basis), 
                              ReLU(), # softplus in the original paper 
                              Linear(n_atom_basis, n_atom_basis),
                              ReLU(), # softplus in the original paper
                              Linear(n_atom_basis, 1)
                              )

    def aggregate(self, message, neighborlist):
        aggregated_edge_feature = torch.cat((message[neighborlist[:,0]], message[neighborlist[:,1]]), 1)
        return aggregated_edge_feature

    def update(self, e):
        return self.mlp(e)


class SchNetConv(MessagePassingModule):

    """The convolution layer with filter. To be merged with GraphConv class.
    
    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(self,
                n_atom_basis,
                n_filters,
                n_gaussians,
                cutoff,
                trainable_gauss,
                mean_pooling=False
    ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict({'message_edge_filter': Sequential(GaussianSmearing(start=0.0,
                                                                             stop=cutoff,
                                                                             n_gaussians=n_gaussians,
                                                                             trainable=trainable_gauss),
                                                          Dense(in_features=n_gaussians,
                                                               out_features=n_gaussians,
                                                               activation=shifted_softplus),
                                                          Dense(in_features=n_gaussians,
                                                               out_features=n_filters,
                                                               activation=None)),
                        'message_node_filter': Dense(in_features=n_atom_basis,
                                                     out_features=n_filters,
                                                     bias=False),
                        'update_function': Sequential(Dense(in_features=n_filters,
                                                             out_features=n_atom_basis,
                                                             activation=shifted_softplus),
                                                      Dense(in_features=n_atom_basis,
                                                             out_features=n_atom_basis,
                                                             activation=None))
                        })


    def message(self, r, e, a):
        """The message function for SchNet convoltuions 
        
        
        Args:
            r (TYPE): node inputs 
            e (TYPE): edge inputs 
            a (TYPE): neighbor list
        
        Returns:
            TYPE: message should a pair of message and 
        """
        # update edge feature 
        e = self.moduledict['message_edge_filter'](e)
        # convection: update 
        r = self.moduledict['message_node_filter'](r)
        # check if the dimensions of r and e are the same 
        assert r.shape[-1] == e.shape[-1]
        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * e  # (ri [] eij) -> rj, []: *, +, (,)
        return message 

    def update(self, r):
        return self.moduledict['update_function'](r)

class GraphAttention(MessagePassingModule):

    """Weighted graph pooling layer based on self attention 
    
    Attributes:
        activation (TYPE): Description
        weight (TYPE): Description
    """
    
    def __init__(self, n_atom_basis):
        super(GraphAttention, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(1, 2 * n_atom_basis))
        self.activation = LeakyReLU()
        
    def message(self, r, e, a):
        """weight_ij is the importance factor of node j to i
           weight_ji is the importance factor of node i to j
        
        Args:
            r (TYPE): Description
            e (TYPE): Description
            a (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # i -> j
        weight_ij = torch.exp(self.activation(torch.cat((r[a[:, 0]], r[a[:, 1]]), dim=1) * \
                             self.weight).sum(-1))
        # j -> i
        weight_ji = torch.exp(self.activation(torch.cat((r[a[:, 1]], r[a[:, 0]]), dim=1) * \
                             self.weight).sum(-1))

        weight_ii = torch.exp(self.activation(torch.cat((r, r), dim=1) * \
                             self.weight).sum(-1))
        
        normalization = scatter_add(weight_ij, a[:, 0], dim_size=r.shape[0]) \
                        + scatter_add(weight_ji, a[:, 1], dim_size=r.shape[0]) + weight_ii

        a_ij = weight_ij/normalization[a[:, 0]] # the importance of node j’s features to node i
        a_ji = weight_ji/normalization[a[:, 1]] # the importance of node i’s features to node j
        a_ii = weight_ii/normalization
        
        message = r[a[:, 0]] * a_ij[:, None], \
                  r[a[:, 1]] * a_ij[:, None], \
                  r * a_ii[:, None]
        
        return message

    def forward(self, r, e, a):
        # Base case
        graph_size = r.shape[0]

        rij, rji, r = self.message(r, e, a)

        # i -> j propagate
        r += self.aggregate(rij, a[:, 1], graph_size)
        # j -> i propagate
        r += self.aggregate(rji, a[:, 0], graph_size)

        r = self.update(r)

        return r

class NodeMultiTaskReadOut(nn.Module):

    """Stack Multi Task outputs

        example multitaskdict:
                {
                "myenergy0":
                            [
                                ['linear', {'in_features': 10, 'out_features': 20}],
                                ['linear', {'in_features': 20, 'out_features': 1}]
                            ],

                "myenergy1": 
                            [
                                ['linear', {'in_features': 10, 'out_features': 20}],
                                ['linear', {'in_features': 20, 'out_features': 1}]
                            ],
                "Muliken charges": 
                            [
                                ['linear', {'in_features': 10, 'out_features': 20}],
                                ['linear', {'in_features': 20, 'out_features': 10}]
                            ]
                }
    """

    def __init__(self, multitaskdict):
        """Summary
        
        Args:
            multitaskdict (TYPE): Description
        """
        super(NodeMultiTaskReadOut, self).__init__()
        # construct moduledict 
        self.readout = construct_ModuleDict(multitaskdict)

    def forward(self, r):
        predict_dict=dict()
        for key in self.readout:
            predict_dict[key] = self.readout[key](r)

        return predict_dict 

class GraphDis(GeometricOperations):

    """Compute distance matrix on the fly 
    
    Attributes:
        box_size (numpy.array): Length of the box, dim = (3, )
        cutoff (float): cutoff for convolution
        F (int): Fr + Fe
        Fe (int): edge feature length
        Fr (int): node feature length
    """
    
    def __init__(
        self,
        Fr,
        Fe,
        cutoff,
        box_size=None
    ):
        super(GraphDis, self).__init__()

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
        
    def forward(self, xyz, bond_adj, bond_len, bond_par):
        e = (
            xyz[bond_adj[:, 0]] - xyz[bond_adj[:, 1]]
        ).pow(2).sum(1).sqrt()[:, None]

        ebond = bond_par * (e - bond_len)**2
        energy = 0.5 * scatter_add(src=ebond, index=bond_adj[:, 0], dim=0, dim_size=xyz.shape[0])
        energy += 0.5 * scatter_add(src=ebond, index=bond_adj[:, 1], dim=0, dim_size=xyz.shape[0])
 
        return ebond

# Test 

class TestModules(unittest.TestCase):

    def testBaseEdgeUpdate(self):
        # initialize basic graphs 
        a = torch.LongTensor([[0, 1], [2,3], [1,3], [4,5], [3,4]])
        e = torch.rand(5, 10)
        r_in = torch.rand(6, 10)
        model = MessagePassingModule()
        r_out = model(r_in, e, a)
        self.assertEqual(r_in.shape, r_out.shape, "The node feature dimensions should be same for the base case")

    def testBaseMessagePassing(self):

        # initialize basic graphs 
        a = torch.LongTensor([[0, 1], [2,3], [1,3], [4,5], [3,4]])
        e_in = torch.rand(5, 10)
        r = torch.rand(6, 10)
        model = EdgeUpdateModule()
        e_out = model(r, e_in, a)
        self.assertEqual(e_in.shape, e_out.shape, "The edge feature dimensions should be same for the base case")

    def testSchNetMPNN(self):
        # contruct a graph 
        a = torch.LongTensor([[0, 1], [2,3], [1,3], [4,5], [3,4]])
        # SchNet params 
        n_atom_basis = 10
        n_filters = 10
        n_gaussians = 10
        num_nodes = 6
        cutoff = 0.5

        e = torch.rand(5, n_atom_basis)
        r_in = torch.rand(num_nodes, n_atom_basis)

        model = SchNetConv(n_atom_basis,
                                n_filters,
                                n_gaussians,
                                cutoff=2.0,
                                trainable_gauss=False,
                                mean_pooling=False)

        r_out = model(r_in, e, a)
        self.assertEqual(r_in.shape, r_out.shape, "The node feature dimensions should be same for the SchNet Convolution case")

    def testSchNetEdgeUpdate(self):
        # contruct a graph 
        a = torch.LongTensor([[0, 1], [2,3], [1,3], [4,5], [3,4]])
        # SchNet params 
        n_atom_basis = 10
        num_nodes = 6

        e_in = torch.rand(5, 1)
        r = torch.rand(num_nodes, n_atom_basis)

        model = SchNetEdgeUpdate(n_atom_basis=n_atom_basis)
        e_out = model(r, e_in, a)

        self.assertEqual(e_in.shape, e_out.shape, "The edge feature dimensions should be same for the SchNet Edge Update case")

    def testGAT(self):
        n_atom_basis= 10

        a = torch.LongTensor([[0, 1], [2,3], [1,3], [4,5], [3,4]])
        e = torch.rand(5, n_atom_basis)
        r_in = torch.rand(6, n_atom_basis)

        attention = GraphAttention(n_atom_basis=n_atom_basis)

        r_out = attention(r_in, e, a)

        self.assertEqual(r_out.shape, r_in.shape)

    def testmultitask(self):

        n_atom = 10 
        r = torch.rand(n_atom, 5)

        multitaskdict = {
                            "myenergy0":
                                        [
                                            ['linear', {'in_features': 5, 'out_features': 20}],
                                            ['linear', {'in_features': 20, 'out_features': 1}]
                                        ],

                            "myenergy1": 
                                        [
                                            ['linear', {'in_features': 5, 'out_features': 20}],
                                            ['linear', {'in_features': 20, 'out_features': 1}]
                                        ],
                            "Muliken charges": 
                                        [
                                            ['linear', {'in_features': 5, 'out_features': 20}],
                                            ['linear', {'in_features': 20, 'out_features': 10}]
                                        ]
                            }

        model = NodeMultiTaskReadOut(multitaskdict)
        output = model(r)

        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    unittest.main()



