from torch import nn
import torch

from nff.nn.models.conformers import WeightedConformers
from nff.nn.modules import (CpSchNetConv, ChemPropMsgToNode,
                            ChemPropInit)
from nff.utils.tools import make_directed


class SchNetFeatures(WeightedConformers):
    def __init__(self, modelparams):

        WeightedConformers.__init__(self, modelparams)

        input_layers = modelparams["input_layers"]
        output_layers = modelparams["output_layers"]

        self.W_i = ChemPropInit(input_layers=input_layers)
        self.convolutions = self.make_convs(modelparams)
        self.W_o = ChemPropMsgToNode(
            output_layers=output_layers)

        self.n_bond_hidden = modelparams["n_bond_hidden"]

    def make_convs(self, modelparams):

        num_conv = modelparams["n_convolutions"]
        same_filters = modelparams["same_filters"]

        convs = nn.ModuleList([CpSchNetConv(**modelparams)
                               for _ in range(num_conv)])
        if same_filters:
            convs = nn.ModuleList([convs[0] for _ in range(num_conv)])

        return convs

    def make_h(self, batch, nbr_list, r):

        """
        Initialize the hidden bond features
        """

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
        h_0 = torch.zeros((nbr_dim,  self.n_bond_hidden))
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

        a, _ = make_directed(batch["nbr_list"])
        r = batch["atom_features"]

        # offsets take care of periodic boundary conditions
        offsets = batch.get("offsets", 0)
        e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
             offsets).pow(2).sum(1).sqrt()[:, None]

        # initialize hidden bond features

        h_0, h_new = [
            self.make_h(batch=batch,
                        nbr_list=a,
                        r=r) for _ in range(2)
        ]

        # update edge features

        for conv in self.convolutions:

            h_new = conv(
                h_0=h_0,
                h_new=h_new,
                all_nbrs=a,
                e=e)

        # convert back to node features

        new_node_feats = self.W_o(r=r,
                                  h=h_new,
                                  nbrs=a,
                                  num_nodes=r.shape[0])

        # TO GET PROPER COMPARISON TO ORIGINAL CHEMPROP,
        # NEED TO CONCATENATE WITH RDKIT 2D FEATURES

        return new_node_feats, xyz
