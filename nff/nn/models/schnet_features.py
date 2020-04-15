from torch import nn
import torch
import copy

from nff.nn.models.schnet import DEFAULT_DROPOUT_RATE
from nff.nn.models.conformers import WeightedConformers
from nff.nn.modules import SchNetFeaturesConv, ChemPropConv
from nff.nn.utils import construct_sequential
from nff.utils.scatter import chemprop_msg_to_node

class SchNetFeatures(WeightedConformers):
    def __init__(self, modelparams):
        WeightedConformers.__init__(self, modelparams)

        n_atom_basis = modelparams["n_atom_basis"]
        n_bond_hidden = modelparams["n_bond_hidden"]
        n_filters = modelparams["n_filters"]
        n_gaussians = modelparams["n_gaussians"]
        gauss_embed = modelparams.get("gauss_embed", True)
        n_convolutions = modelparams["n_convolutions"]
        cutoff = modelparams["cutoff"]
        trainable_gauss = modelparams["trainable_gauss"]
        dropout_rate = modelparams.get("dropout_rate",
                                       DEFAULT_DROPOUT_RATE)
        edge_init_layers = modelparams["edge_init"]

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

        self.edge_init = construct_sequential(
            edge_init_layers)
        self.conv = ChemPropConv(n_bond_hidden=n_bond_hidden,
                                 dropout_rate=dropout_rate)

        self.num_conv = n_convolutions
        self.n_bond_hidden = n_bond_hidden

        delattr(self, "atom_embed")

    def make_bond_feats(self, batch, nbr_list, r):

        # directed neighbour list

        bond_list = batch["bonded_nbr_list"]
        bond_feats = batch["bond_features"]

        nbr_dim = nbr_list.shape[0]
        # nbr_dim = ordered_a.shape[0]

        new_feats = torch.zeros((nbr_dim,  self.n_bond_hidden))
        new_feats = new_feats.to(bond_list.device)

        # feats_ij = torch.zeros((nbr_dim,  self.n_bond_hidden))
        # feats_ij = feats_ij.to(bond_list.device)

        # feats_ji = torch.zeros((nbr_dim,  self.n_bond_hidden))
        # feats_ji = feats_ji.to(bond_list.device)

        bond_idx = (bond_list[:, None] == nbr_list
                    ).prod(-1).nonzero()[:, 1]

        # bond_idx_ij = (bond_list[:, None] == nbr_list
        #                ).prod(-1).nonzero()[:, 1]
        # bond_idx_ji = (bond_list.flip(1)[:, None] == nbr_list
        #                ).prod(-1).nonzero()[:, 1]

        # this will take into account both i < j and
        # i > j because nbr_list is directed

        cat_feats = torch.cat((r[bond_list[:, 0]], bond_feats),
                              dim=1)
        hidden_feats = self.edge_init(cat_feats)

        new_feats[bond_idx] = hidden_feats

        return new_feats

        # feats_ij[bond_idx] = hidden_ij
        # feats_ji[bond_idx] = hidden_ji

        # return feats_ij, feats_ji

    def convolve(self, batch, xyz=None, xyz_grad=False):

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]

        if xyz_grad:
            xyz.requires_grad = True

        # a = batch["nbr_list"]
        a = torch.cat((batch["bonded_nbr_list"],
                       batch["bonded_nbr_list"].flip(1)))
        # mask = a[:, 0] < a[:, 1]
        # ordered_a = a[mask]

        # offsets take care of periodic boundary conditions
        # offsets = batch.get("offsets", 0)
        # e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
        #      offsets).pow(2).sum(1).sqrt()[:, None]

        # initialize with atom features from graph
        r = copy.deepcopy(batch["atom_features"])

        # get bond features from graph
        init_bond_feats = self.make_bond_feats(batch=batch,
                                          nbr_list=a,
                                          r=r)
        new_bond_feats = self.make_bond_feats(batch=batch,
                                          nbr_list=a,
                                          r=r)

        for i in range(self.num_conv):
            # for i, conv in enumerate(self.convolutions):
            conv = self.conv

            new_bond_feats = conv(
                init_bond_feats=init_bond_feats, 
                bond_feats=new_bond_feats, 
                bond_nbrs=a)

            # dr = conv(r=r, e=e, a=ordered_a,
            #           bond_ij=bond_ij, bond_ji=bond_ji)
            # r = r + dr
            
        new_node_feats = chemprop_msg_to_node(h=new_bond_feats, 
            nbrs=a, 
            num_nodes=r.shape[0])

        return new_node_feats, xyz
