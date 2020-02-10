import numpy as np

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, ModuleDict

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.activations import shifted_softplus
from nff.nn.graphconv import MessagePassingModule, EdgeUpdateModule
from nff.nn.utils import construct_sequential, construct_module_dict
from nff.utils.scatter import compute_grad

import unittest
import itertools
import copy

EPSILON = 1e-15

DEFAULT_BONDPRIOR_PARAM = {'k': 20.0}


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
            ReLU(),  # softplus in the original paper
            Linear(n_atom_basis, n_atom_basis),
            ReLU(),  # softplus in the original paper
            Linear(n_atom_basis, 1)
        )

    def aggregate(self, message, neighborlist):
        aggregated_edge_feature = torch.cat((message[neighborlist[:, 0]], message[neighborlist[:, 1]]), 1)
        return aggregated_edge_feature

    def update(self, e):
        return self.mlp(e)


class SchNetConv(MessagePassingModule):

    """The convolution layer with filter.
    
    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(self,
                 n_atom_basis,
                 n_filters,
                 n_gaussians,
                 cutoff,
                 trainable_gauss,
                 ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict({
            'message_edge_filter': Sequential(
                GaussianSmearing(
                    start=0.0,
                    stop=cutoff,
                    n_gaussians=n_gaussians,
                    trainable=trainable_gauss
                ),
                Dense(in_features=n_gaussians, out_features=n_gaussians),
                shifted_softplus(),
                Dense(in_features=n_gaussians, out_features=n_filters)
            ),
            'message_node_filter': Dense(in_features=n_atom_basis, out_features=n_filters),
            'update_function': Sequential(
                Dense(in_features=n_filters, out_features=n_atom_basis),
                shifted_softplus(),
                Dense(in_features=n_atom_basis, out_features=n_atom_basis)
            )
        })

    def message(self, r, e, a, aggr_wgt=None):
        """The message function for SchNet convoltuions 
        Args:
            r (TYPE): node inputs
            e (TYPE): edge inputs
            a (TYPE): neighbor list
            aggr_wgt (None, optional): Description

        Returns:
            TYPE: message should a pair of message and
        """
        # update edge feature
        e = self.moduledict['message_edge_filter'](e)
        # convection: update
        r = self.moduledict['message_node_filter'](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

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

        a_ij = weight_ij / normalization[a[:, 0]]  # the importance of node j’s features to node i
        a_ji = weight_ji / normalization[a[:, 1]]  # the importance of node i’s features to node j
        a_ii = weight_ii / normalization  # self-attention

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

        multitaskdict = {
            'myenergy_0': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'myenergy_1': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ],
            'muliken_charges': [
                {'name': 'linear', 'param' : { 'in_features': 5, 'out_features': 20}},
                {'name': 'linear', 'param' : { 'in_features': 20, 'out_features': 1}}
            ]
        }

        example post_readout:

        def post_readout(predict_dict, readoutdict):
            sorted_keys = sorted(list(readoutdict.keys()))
            sorted_ens = torch.sort(torch.stack([predict_dict[key] for key in sorted_keys]))[0]
            sorted_dic = {key: val for key, val in zip(sorted_keys, sorted_ens) }
            return sorted_dic
    """

    def __init__(self, multitaskdict, post_readout=None):
        """Summary

        Args:
            multitaskdict (dict): dictionary that contains model information
        """
        super(NodeMultiTaskReadOut, self).__init__()
        # construct moduledict
        self.readout = construct_module_dict(multitaskdict)
        self.post_readout = post_readout
        self.multitaskdict = multitaskdict

    def forward(self, r):
        predict_dict = dict()
        for key in self.readout:
            predict_dict[key] = self.readout[key](r)
        if self.post_readout is not None:
            predict_dict = self.post_readout(predict_dict, self.multitaskdict)

        return predict_dict


class BondPrior(torch.nn.Module):

    def __init__(self, modelparams=DEFAULT_BONDPRIOR_PARAM):
        torch.nn.Module.__init__(self)
        self.k = modelparams['k']
        
    def forward(self, batch):
        
        result = {}
        
        num_bonds = batch["num_bonds"].tolist()
        
        xyz = batch['nxyz'][:, 1:4]
        xyz.requires_grad = True
        bond_list = batch["bonds"]
        
        r_0 = batch['bond_len'].squeeze()
        
        r = (xyz[bond_list[:, 0]] - xyz[bond_list[:, 1]]).pow(2).sum(-1).sqrt()
        
        e = self.k * ( r - r_0).pow(2)
        
        E = torch.stack([e.sum(0) for e in torch.split(e, num_bonds)])
        
        result['energy'] = E.sum()
        result['energy_grad'] = compute_grad(inputs=xyz, output=E)
        
        return result


# Test

class TestModules(unittest.TestCase):

    def testBaseEdgeUpdate(self):
        # initialize basic graphs

        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e = torch.rand(5, 10)
        r_in = torch.rand(6, 10)
        model = MessagePassingModule()
        r_out = model(r_in, e, a)
        self.assertEqual(r_in.shape, r_out.shape, "The node feature dimensions should be same for the base case")

    def testBaseMessagePassing(self):
        # initialize basic graphs
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e_in = torch.rand(5, 10)
        r = torch.rand(6, 10)
        model = EdgeUpdateModule()
        e_out = model(r, e_in, a)
        self.assertEqual(e_in.shape, e_out.shape, "The edge feature dimensions should be same for the base case")

    def testSchNetMPNN(self):
        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        # SchNet params
        n_atom_basis = 10
        n_filters = 10
        n_gaussians = 10
        num_nodes = 6
        cutoff = 0.5

        e = torch.rand(5, n_atom_basis)
        r_in = torch.rand(num_nodes, n_atom_basis)

        model = SchNetConv(
            n_atom_basis,
            n_filters,
            n_gaussians,
            cutoff=2.0,
            trainable_gauss=False,
        )

        r_out = model(r_in, e, a)
        self.assertEqual(r_in.shape, r_out.shape,
                         "The node feature dimensions should be same.")

    def testDoubleNodeConv(self):

        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        num_nodes = 6
        num_features = 12

        update_layers =  [{'name': 'linear', 'param' : {'in_features': 2*num_features,
                                                        'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}},
                          {'name': 'linear', 'param' : {'in_features': num_features,
                                                  'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}}]


        r_in = torch.rand(num_nodes, num_features)
        model = DoubleNodeConv(update_layers)
        r_out = model(r=r_in, e=None, a=a)

        self.assertEqual(r_in.shape, r_out.shape,
                         "The node feature dimensions should be same.")

    def testSingleNodeConv(self):

        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        num_nodes = 6
        num_features = 12

        update_layers =  [{'name': 'linear', 'param' : {'in_features': num_features,
                                                        'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}},
                          {'name': 'linear', 'param' : {'in_features': num_features,
                                                  'out_features': num_features}},
                          {'name': 'shifted_softplus', 'param': {}}]


        r_in = torch.rand(num_nodes, num_features)
        model = SingleNodeConv(update_layers)
        r_out = model(r=r_in, e=None, a=a)

        self.assertEqual(r_in.shape, r_out.shape,
                         "The node feature dimensions should be same for the SchNet Convolution case")

    def testSchNetEdgeUpdate(self):
        # contruct a graph
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        # SchNet params
        n_atom_basis = 10
        num_nodes = 6

        e_in = torch.rand(5, 1)
        r = torch.rand(num_nodes, n_atom_basis)

        model = SchNetEdgeUpdate(n_atom_basis=n_atom_basis)
        e_out = model(r, e_in, a)

        self.assertEqual(e_in.shape, e_out.shape,
                         "The edge feature dimensions should be same for the SchNet Edge Update case")

    def testGAT(self):
        n_atom_basis = 10

        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
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
                    {'name': 'Dense', 'param': {'in_features': 5, 'out_features': 20}},
                    {'name': 'shifted_softplus', 'param': {}},
                    {'name': 'Dense', 'param': {'in_features': 20, 'out_features': 1}}
                ],

            "myenergy1":
                [
                    {'name': 'linear', 'param': {'in_features': 5, 'out_features': 20}},
                    {'name': 'Dense', 'param': {'in_features': 20, 'out_features': 1}}
                ],
            "Muliken charges":
                [
                    {'name': 'linear', 'param': {'in_features': 5, 'out_features': 20}},
                    {'name': 'linear', 'param': {'in_features': 20, 'out_features': 1}}
                ]
        }

        model = NodeMultiTaskReadOut(multitaskdict)
        output = model(r)


if __name__ == '__main__':
    unittest.main()



