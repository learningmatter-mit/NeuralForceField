
import torch
import torch.nn as nn
from torch.nn import (Sequential, Linear, ReLU, LeakyReLU,
                      ModuleDict, Softmax)

from torch.nn.functional import softmax

from nff.nn.layers import Dense, GaussianSmearing
from nff.utils.scatter import scatter_add
from nff.nn.utils import chemprop_msg_update, chemprop_msg_to_node, remove_bias
from nff.nn.activations import shifted_softplus
from nff.utils.tools import layer_types
from nff.nn.graphconv import (
    MessagePassingModule,
    EdgeUpdateModule,
)
from nff.nn.utils import construct_sequential, construct_module_dict
from nff.utils.scatter import compute_grad

import unittest


EPSILON = 1e-15

DEFAULT_BONDPRIOR_PARAM = {"k": 20.0}


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
            Linear(n_atom_basis, 1),
        )

    def aggregate(self, message, neighborlist):
        aggregated_edge_feature = torch.cat(
            (message[neighborlist[:, 0]], message[neighborlist[:, 1]]), 1
        )
        return aggregated_edge_feature

    def update(self, e):
        return self.mlp(e)


class SchNetEdgeFilter(nn.Module):
    def __init__(self,
                 cutoff,
                 n_gaussians,
                 trainable_gauss,
                 n_filters,
                 dropout_rate,
                 activation='shifted_softplus'):

        super(SchNetEdgeFilter, self).__init__()

        self.filter = Sequential(
            GaussianSmearing(
                start=0.0,
                stop=cutoff,
                n_gaussians=n_gaussians,
                trainable=trainable_gauss,
            ),
            Dense(
                in_features=n_gaussians,
                out_features=n_gaussians,
                dropout_rate=dropout_rate,
            ),
            layer_types[activation](),
            Dense(
                in_features=n_gaussians,
                out_features=n_filters,
                dropout_rate=dropout_rate,
            ))

    def forward(self, e):
        return self.filter(e)


class SchNetConv(MessagePassingModule):

    """The convolution layer with filter.

    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(
        self,
        n_atom_basis,
        n_filters,
        n_gaussians,
        cutoff,
        trainable_gauss,
        dropout_rate,
    ):
        super(SchNetConv, self).__init__()
        self.moduledict = ModuleDict(
            {
                "message_edge_filter": SchNetEdgeFilter(
                    cutoff=cutoff,
                    n_gaussians=n_gaussians,
                    trainable_gauss=trainable_gauss,
                    n_filters=n_filters,
                    dropout_rate=dropout_rate),

                "message_node_filter": Dense(
                    in_features=n_atom_basis,
                    out_features=n_filters,
                    dropout_rate=dropout_rate,
                ),
                "update_function": Sequential(
                    Dense(
                        in_features=n_filters,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                ),
            }
        )

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
        e = self.moduledict["message_edge_filter"](e)
        # convection: update
        r = self.moduledict["message_node_filter"](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * \
            e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)


class MixedSchNetConv(MessagePassingModule):

    """The convolution layer with filter.

    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(
        self,
        n_atom_basis,
        n_filters,
        dropout_rate,
        n_bond_hidden,
        activation='shifted_softplus'
    ):
        super(MixedSchNetConv, self).__init__()
        self.moduledict = ModuleDict(
            {

                "message_node_filter": Dense(
                    in_features=n_atom_basis,
                    out_features=(n_filters + n_bond_hidden),
                    dropout_rate=dropout_rate,
                ),
                "update_function": Sequential(
                    Dense(
                        in_features=(n_filters + n_bond_hidden),
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                    layer_types[activation](),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                        dropout_rate=dropout_rate,
                    ),
                ),
            }
        )

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
        # convection: update
        r = self.moduledict["message_node_filter"](r)

        # soft aggr if aggr_wght is provided
        if aggr_wgt is not None:
            r = r * aggr_wgt

        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * \
            e  # (ri [] eij) -> rj, []: *, +, (,)
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)


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
        weight_ij = torch.exp(
            self.activation(
                torch.cat((r[a[:, 0]], r[a[:, 1]]), dim=1) * self.weight
            ).sum(-1)
        )
        # j -> i
        weight_ji = torch.exp(
            self.activation(
                torch.cat((r[a[:, 1]], r[a[:, 0]]), dim=1) * self.weight
            ).sum(-1)
        )

        weight_ii = torch.exp(
            self.activation(torch.cat((r, r), dim=1) * self.weight).sum(-1)
        )

        normalization = (
            scatter_add(weight_ij, a[:, 0], dim_size=r.shape[0])
            + scatter_add(weight_ji, a[:, 1], dim_size=r.shape[0])
            + weight_ii
        )

        a_ij = (
            weight_ij / normalization[a[:, 0]]
        )  # the importance of node j’s features to node i
        a_ji = (
            weight_ji / normalization[a[:, 1]]
        )  # the importance of node i’s features to node j
        a_ii = weight_ii / normalization  # self-attention

        message = (
            r[a[:, 0]] * a_ij[:, None],
            r[a[:, 1]] * a_ji[:, None],
            r * a_ii[:, None],
        )

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


class ConfAttention(nn.Module):
    """
    Module to apply an attention mechanism to different conformers
    of a species to learn their respective contribution to the total
    fingerprint.
    """

    def __init__(self,
                 mol_basis,
                 boltz_basis,
                 final_act,
                 equal_weights=False,
                 prob_func='softmax'):
        """
        Args:
            mol_basis (int): dimension of the molecular fingerprint
            boltz_basis (int): dimension into which we embed the boltzmann
                weight as a vector.
            final_act (str): name of the final nonlinear layer to apply to
                the fingerprint.
            equal_weights (bool): whether to not use attention and just use
                equal weights for each conformer.
            prob_func (str): fucntion to use to convert alpha_ij to probabilities
                (i.e. weights that sum to 1).
        Returns:
            None
        """

        super(ConfAttention, self).__init__()

        """
        Xavier initializations from
        https://github.com/Diego999/pyGAT/blob/master/layers.py
        """

        # if boltz_basis = None then don't embed the boltmzann vector
        if boltz_basis is None:
            self.embed_boltz = False

        # otherwise initialize a layer to embed it and `fp_linear` to
        # convert cat([mol_fp, boltz_fp]) to a new vector of dimension
        # `mol_basis`.

        else:
            self.embed_boltz = True
            self.boltz_lin = torch.nn.Linear(1, boltz_basis)
            self.boltz_act = Softmax(dim=1)

            self.fp_linear = torch.nn.Linear(
                mol_basis + boltz_basis, mol_basis, bias=False)

        self.equal_weights = equal_weights

        # if you don't want equal weights, then initialize `att_weight`,
        # which will be used to get the weight og each conformer.

        if not self.equal_weights:
            self.att_weight = torch.nn.Parameter(torch.rand(1, 2 * mol_basis))
            nn.init.xavier_uniform_(self.att_weight, gain=1.414)

        # linear layer
        self.W = torch.nn.Linear(in_features=mol_basis,
                                 out_features=mol_basis)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)

        self.final_act = layer_types[final_act]()
        self.activation = LeakyReLU(inplace=False)
        self.prob_func = prob_func

    def forward(self,
                conf_fps,
                boltzmann_weights):
        """
        Args:
            conf_fps (torch.Tensor): conformer fingerprints for a species
            boltzmann_weights (torch.Tensor): their statistical Boltzmann
                weights.
        Returns:
            output (torch.Tensor): attention-pooled conformer fingerprints
            alpha_ij (torch.Tensor): the learned weight factors between
                each pair of conformers i and j.
        """

        # for backwards compatibility
        if not hasattr(self, "embed_boltz"):
            self.embed_boltz = True
        if not hasattr(self, "equal_weights"):
            self.equal_weights = False
        if not hasattr(self, "prob_func"):
            self.prob_func = 'softmax'

        if self.embed_boltz:
            # increase dimensionality of Boltzmann weight
            boltz_vec = self.boltz_act(self.boltz_lin(boltzmann_weights))

            # concatenate fingerprints with Boltzmann vector
            # and apply linear layer to reduce back to size `mol_basis`
            cat_fps = torch.cat([conf_fps, boltz_vec], dim=1)
            new_fps = self.fp_linear(cat_fps)
        else:
            new_fps = conf_fps

        # directed "neighbor list" that links every fingerprint to each other
        a = torch.LongTensor([[i, j] for i in range(new_fps.shape[0])
                              for j in range(new_fps.shape[0])]
                             ).to(conf_fps.device)

        # make cat(h_i, h_j)
        cat_ij = torch.cat((self.W(new_fps[a[:, 0]]),
                            self.W(new_fps[a[:, 1]])), dim=1)

        # number of conformers
        n_confs = new_fps.shape[0]

        # if equal weights then set all alpha_ij equal
        if self.equal_weights:
            alpha_ij = torch.ones(n_confs * n_confs, 1).to(new_fps.device)
            alpha_ij /= alpha_ij.sum()

        # otherwise proceed with regular attention

        else:
            output = self.activation(
                torch.matmul(
                    self.att_weight, cat_ij.transpose(0, 1)
                )
            )

            if self.prob_func == 'softmax':
                alpha_ij = softmax(output.reshape(n_confs, n_confs),
                                   dim=1).reshape(-1, 1) / n_confs

            elif self.prob_func == 'square':
                out_reshape = (output ** 2).reshape(n_confs, n_confs)
                norm = out_reshape.sum(dim=1).reshape(-1, 1)
                non_norm_prob = out_reshape / norm
                alpha_ij = non_norm_prob.reshape(-1, 1) / n_confs

        fp_j = new_fps[a[:, 1]]
        prod = alpha_ij * self.W(fp_j)

        # sum over neighbors and over fingerprints
        summed_fps = prod.sum(0)

        # put through nonlinearity
        output = self.final_act(summed_fps)

        return output, alpha_ij


class LinearConfAttention(ConfAttention):
    """
    Similar to ConfAttention, but instead of concatenating each pair
    of fingerprints to learn the importance of one to the other, 
    just uses each fingerprint alone to learn its importance.
    """

    def __init__(self,
                 mol_basis,
                 boltz_basis,
                 final_act,
                 equal_weights=False,
                 prob_func='softmax'):
        """
        Args:
            mol_basis (int): dimension of the molecular fingerprint
            boltz_basis (int): dimension into which we embed the boltzmann
                weight as a vector.
            final_act (str): name of the final nonlinear layer to apply to
                the fingerprint.
            equal_weights (bool): whether to not use attention and just use
                equal weights for each conformer.
            prob_func (str): fucntion to use to convert alpha_ij to probabilities
                (i.e. weights that sum to 1).
        Returns:
            None
        """

        super(LinearConfAttention, self).__init__(mol_basis,
                                                  boltz_basis,
                                                  final_act,
                                                  equal_weights)

        # has dimension mol_basis instead of 2 * mol_basis because we're not
        # comparing fingerprint pairs

        if not self.equal_weights:
            self.att_weight = torch.nn.Parameter(torch.rand(1, mol_basis))
            nn.init.xavier_uniform_(self.att_weight, gain=1.414)

        self.prob_func = prob_func

    def forward(self,
                conf_fps,
                boltzmann_weights):
        """
        Args:
            conf_fps (torch.Tensor): conformer fingerprints for a species
            boltzmann_weights (torch.Tensor): their statistical Boltzmann
                weights.
        Returns:
            output (torch.Tensor): attention-pooled conformer fingerprints
            alpha_i (torch.Tensor): the learned weight factors for
                each conformer i.
        """

        # for backwards compatibility
        if not hasattr(self, "embed_boltz"):
            self.embed_boltz = True
        if not hasattr(self, "equal_weights"):
            self.equal_weights = False
        if not hasattr(self, "prob_func"):
            self.prob_func = 'softmax'

        if self.embed_boltz:
            # increase dimensionality of Boltzmann weight
            boltz_vec = self.boltz_act(self.boltz_lin(boltzmann_weights))

            # concatenate fingerprints with Boltzmann vector
            # and apply linear layer to reduce back to size `mol_basis`
            cat_fps = torch.cat([conf_fps, boltz_vec], dim=1)
            new_fps = self.fp_linear(cat_fps)
        else:
            new_fps = conf_fps

        n_confs = new_fps.shape[0]

        # if equal weights then set all alpha_ij equal

        if self.equal_weights:
            alpha_i = torch.ones(n_confs).to(new_fps.device).reshape(-1, 1)
            alpha_i /= alpha_i.sum()

        # otherwise apply attention to each fingerprint individually

        else:
            output = self.activation(
                torch.matmul(
                    self.att_weight, self.W(new_fps).transpose(0, 1)
                )
            )

            if self.prob_func == 'softmax':
                alpha_i = softmax(output, dim=1).reshape(-1, 1)
            elif self.prob_func == 'square':
                alpha_i = (output ** 2 / (output ** 2).sum()).reshape(-1, 1)

        prod = alpha_i * self.W(new_fps)

        # sum over neighbors and over fingerprints
        summed_fps = prod.sum(0)

        # put through nonlinearity
        output = self.final_act(summed_fps)

        return output, alpha_i


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
        self.k = modelparams["k"]

    def forward(self, batch):

        result = {}

        num_bonds = batch["num_bonds"].tolist()

        xyz = batch["nxyz"][:, 1:4]
        xyz.requires_grad = True
        bond_list = batch["bonds"]

        r_0 = batch["bond_len"].squeeze()

        r = (xyz[bond_list[:, 0]] - xyz[bond_list[:, 1]]).pow(2).sum(-1).sqrt()

        e = self.k * (r - r_0).pow(2)

        E = torch.stack([e.sum(0) for e in torch.split(e, num_bonds)])

        result["energy"] = E.sum()
        result["energy_grad"] = compute_grad(inputs=xyz, output=E)

        return result


class ChemPropConv(MessagePassingModule):

    """
    ChemProp convolution module.
    """

    def __init__(self,
                 n_edge_hidden,
                 dropout_rate,
                 activation,
                 **kwargs):
        """
        Args:
            n_edge_hidden: dimension of the hidden edge vector
            dropout_rate (float): dropout rate
            activation (str): name of non-linear activation function
        Returns:
            None
        """

        MessagePassingModule.__init__(self)

        # As in the original ChemProp paper,
        # the features are added together linearly
        # with no bias. This means that features
        # equal to 0 don't contribute to the output.

        # This is important, for example, for
        # CpSchNetConv, in which every non-
        # bonded neighbour has zeros for its
        # ChemProp bond features. We don't want
        # these zeros contributing to the output.

        self.dense = Dense(
            in_features=n_edge_hidden,
            out_features=n_edge_hidden,
            dropout_rate=dropout_rate,
            bias=False
        )
        self.activation = layer_types[activation]()

    def message(self, h_new, nbrs):
        """
        Get the chemprop MPNN message.
        Args:
            h_new (torch.Tensor): new hidden edge vector
            nbrs (torch.LongTensor): bonded neighbor list
        Returns:
            msg (torch.Tensor): message from nearby atoms
        """

        msg = chemprop_msg_update(h=h_new,
                                  nbrs=nbrs)
        return msg

    def update(self, msg, h_0):
        """
        Update the hiddn edge features.
        Args:
            msg (torch.Tensor): message vector
            h_0 (torch.Tensor): initial hidden edge
                vector.
        Returns:
            update_feats (torch.Tensor): updated
                features after convolution.
        """

        # addition of original edge features with
        # `msg` after going through a dense layer
        add_feats = h_0 + self.dense(msg)
        # apply the nonlinearity
        update_feats = self.activation(add_feats)

        return update_feats

    def forward(self, h_0, h_new, nbrs):
        """
        Apply a convolution layer.
        Args:
            h_0 (torch.Tensor): initial hidden edge
                vector.
            h_new (torch.Tensor): latest hidden edge vector
            nbrs (torch.LongTensor): bonded neighbor list
        Returns:
            update_feats (torch.Tensor): updated
                features after convolution.
        """

        msg = self.message(h_new=h_new,
                           nbrs=nbrs)
        update_feats = self.update(msg=msg,
                                   h_0=h_0)

        return update_feats


class CpSchNetConv(ChemPropConv):

    """
    Module for mixing the ChemProp convolution with a distance-based
    SchNet convolution.
    """

    def __init__(
        self,
        n_bond_hidden,
        cp_dropout,
        gauss_embed,
        cutoff,
        n_gaussians,
        trainable_gauss,
        n_filters,
        schnet_dropout,
        activation,
        **kwargs

    ):
        """
        Args:
            n_bond_hidden (int): dimensionality of the hideen bond features
            cp_dropout (float): dropout rate for the ChemProp convolutions
            gauss_embed (bool): whether to embed the distances in a Gaussian
                basis.
            cutoff (float): distance cutoff for the 3D convolutions.
            n_gaussians (int): number of Gaussians in which to expand the 
                distances.
            trainable_gauss (bool): whether the parameters of the Gaussians
                (width and center locations) are learnable parameters.
            n_filters (int): dimensionality of the 3D edge feature
            schnet_dropout (float): dropout rate for the SchNet convolutions
            activation (str): name of non-linear activation function

        Returns:
            None
        """

        ChemPropConv.__init__(self,
                              n_edge_hidden=n_bond_hidden,
                              dropout_rate=cp_dropout,
                              activation=activation)

        self.n_bond_hidden = n_bond_hidden
        self.moduledict = ModuleDict({})

        schnet_hidden_dim = n_filters if (gauss_embed) else 1

        edge_filter = Sequential(
            GaussianSmearing(
                start=0.0,
                stop=cutoff,
                n_gaussians=n_gaussians,
                trainable=trainable_gauss,
            ),
            Dense(
                in_features=n_gaussians,
                out_features=n_filters,
                dropout_rate=schnet_dropout,
            ),
            layer_types[activation]())

        # layers to apply to schnet edge vectors after they've been
        # updated

        schnet_layers = Sequential(
            Dense(
                in_features=schnet_hidden_dim,
                out_features=schnet_hidden_dim,
                dropout_rate=schnet_dropout,
            ),
            layer_types[activation]())

        self.moduledict.update({"edge_filter": edge_filter,
                                "schnet_layers": schnet_layers})
        if not gauss_embed:
            self.moduledict.pop("edge_filter")

        self.update_schnet = True

    def schnet_msg(self,
                   e,
                   nbr_list):

        # this doesn't work because of the indexing --
        # nbr_list = [0, 1] has indices corresponding to
        # atoms, not to the edge vector.

        # if we really want to do this then we have to go
        # full-blown chemprop on all of it. Which could be
        # very slow, although if we have `bond_idx` and a
        # larger sub-batch size it could be doable?

        # What's the alternative? A SchNet atom feature-based
        # update? Probably makes the most sense.

        if self.update_schnet:
            # add edge features of anything that is in
            # another's neighbor list
            e[nbr_list[:, 0]] += e[nbr_list[:, 1]]

            # apply nonlinearities
            e = self.moduledict["schnet_layers"](e)

        return e

    def add_schnet_feats(self,
                         h_schnet,
                         h_new,
                         nbr_list):
        """
        Add SchNet edge features to the updated hidden bond features.
        Args:
            h_schnet (torch.Tensor): edge features based on distances
            h_new (torch.Tensor): new hidden bond features
            nbr_list (torch.LongTensor): neighbor list
        Returns:
            new_msg (torch.Tesnor): combination of distance and bond
                features.
        """

        e = self.schnet_msg(e=e,
                            nbr_list=nbr_list)
        new_msg = torch.cat((h_new, e), dim=1)

        return new_msg

    def forward(self,
                h_0,
                h_new,
                all_nbrs,
                bond_nbrs,
                bond_idx):
        """
        Call the module.
        Args:
            h_0 (torch.Tensor): initial hidden edge
                vector.
            h_new (torch.Tensor): latest hidden edge vector
            all_nbrs (torch.LongTensor): 3D neighbor list
            bond_nbrs (torch.LongTensor): bonded neighbor list
            bond_idx (torch.LongTensor): indices that map
                an element of `bond_nbrs` to the corresponding
                element in `nbr_list`. 

        Returns:
            final_h (torch.Tensor): updated edge features after
                onvolution.
        """

        # for backward compatability
        if not hasattr(self, "update_schnet"):
            self.update_schnet = False

        if self.update_schnet:
            cp_msg = self.message(h_new=h_new,
                                  nbrs=all_nbrs)
            final_h = self.update(msg=cp_msg,
                                  h_0=h_0)
            return final_h

        # extract the ChemProp bond features from the complete set
        # of features

        cp_h = h_new[:, :self.n_bond_hidden][bond_idx]
        h0_bond = h_0[bond_idx]

        # get the ChemProp message and update the hidden bond vector
        # with this message

        cp_msg = self.message(h_new=cp_h,
                              nbrs=bond_nbrs)
        h_new_bond = self.update(msg=cp_msg,
                                 h_0=h0_bond)

        # make the entire hiddne edge vector by putting zeros for
        # every pair of atoms that isn't bonded

        nbr_dim = all_nbrs.shape[0]
        h_new = torch.zeros((nbr_dim,  self.n_bond_hidden))
        h_new = h_new.to(bond_idx.device)
        h_new[bond_idx] = h_new_bond

        # extract the SchNet edge features from the complete set
        # of features

        h_schnet = h_new[:, self.n_bond_hidden, :]

        # add the SchNet features

        final_h = self.add_schnet_feats(h_schnet=h_schnet,
                                        h_new=h_new,
                                        nbr_list=all_nbrs)

        return final_h


class ChemPropMsgToNode(nn.Module):
    def __init__(self, output_layers):
        nn.Module.__init__(self)

        # remove bias from linear layers if there

        new_layers = remove_bias(output_layers)
        self.output = construct_sequential(new_layers)

    def forward(self, r, h, nbrs):
        num_nodes = r.shape[0]
        msg_to_node = chemprop_msg_to_node(h=h,
                                           nbrs=nbrs,
                                           num_nodes=num_nodes)
        cat_node = torch.cat((r, msg_to_node), dim=1)
        new_node_feats = self.output(cat_node)

        return new_node_feats


class ChemPropInit(nn.Module):

    def __init__(self, input_layers):
        nn.Module.__init__(self)

        # remove bias from linear layers if there

        new_layers = remove_bias(input_layers)
        self.input = construct_sequential(new_layers)

    def forward(self, r, bond_feats, bond_nbrs):
        cat_feats = torch.cat((r[bond_nbrs[:, 0]], bond_feats),
                              dim=1)
        hidden_feats = self.input(cat_feats)

        return hidden_feats


# Test


class TestModules(unittest.TestCase):
    def testBaseEdgeUpdate(self):
        # initialize basic graphs

        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e = torch.rand(5, 10)
        r_in = torch.rand(6, 10)
        model = MessagePassingModule()
        r_out = model(r_in, e, a)
        self.assertEqual(
            r_in.shape,
            r_out.shape,
            "The node feature dimensions should be same for the base case",
        )

    def testBaseMessagePassing(self):
        # initialize basic graphs
        a = torch.LongTensor([[0, 1], [2, 3], [1, 3], [4, 5], [3, 4]])
        e_in = torch.rand(5, 10)
        r = torch.rand(6, 10)
        model = EdgeUpdateModule()
        e_out = model(r, e_in, a)
        self.assertEqual(
            e_in.shape,
            e_out.shape,
            "The edge feature dimensions should be same for the base case",
        )

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
            n_atom_basis, n_filters, n_gaussians,
            cutoff=2.0, trainable_gauss=False,
        )

        r_out = model(r_in, e, a)
        self.assertEqual(
            r_in.shape, r_out.shape,
            "The node feature dimensions should be same."
        )

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

        self.assertEqual(
            e_in.shape,
            e_out.shape,
            ("The edge feature dimensions should be same"
             "for the SchNet Edge Update case"),
        )

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
            "myenergy0": [
                {"name": "Dense", "param": {"in_features": 5,
                                            "out_features": 20}},
                {"name": "shifted_softplus", "param": {}},
                {"name": "Dense", "param": {"in_features": 20,
                                            "out_features": 1}},
            ],
            "myenergy1": [
                {"name": "linear", "param": {"in_features": 5,
                                             "out_features": 20}},
                {"name": "Dense", "param": {"in_features": 20,
                                            "out_features": 1}},
            ],
            "Muliken charges": [
                {"name": "linear", "param": {"in_features": 5,
                                             "out_features": 20}},
                {"name": "linear", "param": {"in_features": 20,
                                             "out_features": 1}},
            ],
        }

        model = NodeMultiTaskReadOut(multitaskdict)
        output = model(r)


if __name__ == "__main__":
    unittest.main()
