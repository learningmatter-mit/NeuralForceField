from torch import nn
import torch
import copy

import pdb


from nff.nn.models.schnet import DEFAULT_DROPOUT_RATE
from nff.nn.models.conformers import WeightedConformers
from nff.nn.modules import (SchNetFeaturesConv, ChemPropConv,
                            ChemPropMsgToNode, ChemPropInit)
from nff.nn.utils import construct_sequential
from nff.utils.scatter import chemprop_msg_to_node
from nff.utils.tools import make_directed


class SchNetFeatures(WeightedConformers):
    def __init__(self, modelparams):
        WeightedConformers.__init__(self, modelparams)

        n_atom_basis = modelparams["n_atom_basis"]
        n_edge_hidden = modelparams["n_edge_hidden"]
        n_filters = modelparams["n_filters"]
        n_gaussians = modelparams["n_gaussians"]
        gauss_embed = modelparams.get("gauss_embed", True)
        n_convolutions = modelparams["n_convolutions"]
        cutoff = modelparams["cutoff"]
        trainable_gauss = modelparams["trainable_gauss"]
        dropout_rate = modelparams.get("dropout_rate",
                                       DEFAULT_DROPOUT_RATE)
        input_layers = modelparams["input_layers"]
        output_layers = modelparams["output_layers"]

        # self.convolutions = nn.ModuleList(
        #     [
        #         SchNetFeaturesConv(
        #             n_atom_basis=n_atom_basis,
        #             n_bond_hidden=n_bond_hidden,
        #             n_filters=n_filters,
        #             n_gaussians=n_gaussians,
        #             cutoff=cutoff,
        #             trainable_gauss=trainable_gauss,
        #             dropout_rate=dropout_rate,
        #             gauss_embed=gauss_embed
        #         )
        #         for _ in range(n_convolutions)
        #     ]
        # )

        # self.convolution = SchNetFeaturesConv(
        #     n_atom_basis=n_atom_basis,
        #     n_bond_hidden=n_bond_hidden,
        #     n_filters=n_filters,
        #     n_gaussians=n_gaussians,
        #     cutoff=cutoff,
        #     trainable_gauss=trainable_gauss,
        #     dropout_rate=dropout_rate,
        #     gauss_embed=gauss_embed
        # )

        self.W_i = ChemPropInit(input_layers=input_layers)
        self.W_h = ChemPropConv(n_edge_hidden=n_edge_hidden,
                                dropout_rate=dropout_rate)
        self.W_o = ChemPropMsgToNode(output_layers=output_layers)

        self.num_conv = n_convolutions
        self.n_edge_hidden = n_edge_hidden

    def make_h(self, batch, nbr_list, r):

        # get the directed bond list and bond features

        bond_list, was_directed = make_directed(batch["bonded_nbr_list"])
        if was_directed:
            bond_feats = batch["bond_features"]
        else:
            bond_feats = torch.cat([batch["bond_features"]] * 2, dim=0)

        # initialize hidden bond features

        h_0_bond = self.W_i(r=r,
                            bond_feats=bond_feats,
                            bond_nbrs=bond_list)

        # initialize `h_0`, the features of all edges
        # (including non-bonded ones), to zero

        nbr_dim = nbr_list.shape[0]
        h_0 = torch.zeros((nbr_dim,  self.n_edge_hidden))
        h_0 = h_0.to(bond_list.device)

        # set the features of bonded edges equal to the bond
        # features

        bond_idx = (bond_list[:, None] == nbr_list
                    ).prod(-1).nonzero()[:, 1]
        h_0[bond_idx] = h_0_bond

        return h_0

    def convolve(self, batch, xyz=None, xyz_grad=False):

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]

        if xyz_grad:
            xyz.requires_grad = True

        # a, _ = make_directed(batch["nbr_list"])
        a, _ = make_directed(batch["bonded_nbr_list"])
        r = batch["atom_features"]

        # offsets take care of periodic boundary conditions
        # offsets = batch.get("offsets", 0)
        # e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
        #      offsets).pow(2).sum(1).sqrt()[:, None]

        # initialize with atom features from graph

        # initialize hidden bond features

        h_0, h_new = [
            self.make_h(batch=batch,
                        nbr_list=a,
                        r=r) for _ in range(2)
        ]

        # update edge features

        for i in range(self.num_conv):

            h_new = self.W_h(
                h_0=h_0,
                h_new=h_new,
                nbrs=a)

        # convert back to node features

        new_node_feats = self.W_o(r=r,
                                  h=h_new,
                                  nbrs=a,
                                  num_nodes=r.shape[0])

        return new_node_feats, xyz
