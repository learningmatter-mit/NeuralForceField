import torch
from torch import nn
from torch.nn import Embedding, Sequential 

from nff.nn.layers import Dense
from nff.nn.modules.dimenet import (EmbeddingBlock, InteractionBlock,
                                    OutputBlock, get_dense)
from nff.nn.layers import DimeNetRadialBasis as RadialBasis
from nff.nn.layers import DimeNetSphericalBasis as SphericalBasis
from nff.utils.scatter import compute_grad
from nff.nn.activations import shifted_softplus
from nff.nn.models import compute_angle
from nff.utils.scatter import scatter_add
from torch.nn import ModuleDict


class ReadoutBlock(nn.Module):
    """
    Block to convert edge messages to both edge and angle fingerprints
    """

    def __init__(self, embed_dim,
                 n_rbf, n_sbf,
                 activation):
        """
        Args:
            embed_dim (int): embedding size
            n_rbf (int): number of radial basis functions
            activation (str): name of activation layer
        Returns:
            None
        """
        super().__init__()

        # dense layer to convert rbf edge representation
        # to dimension embed_dim
        self.edge_dense = get_dense(n_rbf,
                                    embed_dim,
                                    activation=None,
                                    bias=False)
        
        # edge dense layers
        self.edge_dense_layers = nn.ModuleList(
            [
                get_dense(embed_dim,
                          embed_dim,
                          activation=activation,
                          bias=True)
                for _ in range(3)
            ])
        # final dense layer without bias or activation
        self.edge_dense_layers.append(get_dense(embed_dim,
                                           embed_dim,
                                           activation=None,
                                           bias=False))


        # dense layer to convert sbf angle representation
        # to dimension embed_dim
        self.angle_dense = get_dense(n_sbf,
                                    embed_dim,
                                    activation=None,
                                    bias=False)
        
        # angle dense layers
        self.angle_dense_layers = nn.ModuleList(
            [
                get_dense(embed_dim,
                          embed_dim,
                          activation=activation,
                          bias=True)
                for _ in range(3)
            ])
        # final dense layer without bias or activation
        self.angle_dense_layers.append(get_dense(embed_dim,
                                           embed_dim,
                                           activation=None,
                                           bias=False))

    def forward(self, m_ji, e_rbf, a_sbf, nbr_list, kj_idx, ji_idx):

        # product of e and m
        edge_prod = self.edge_dense(e_rbf) * m_ji

        # Apply the edge dense layers
        edge_feats = edge_prod
        for edge_dense_layer in self.edge_dense_layers:
            edge_feats = edge_dense_layer(edge_feats)

        angle_prod = self.angle_dense(a_sbf) * (m_ji[kj_idx] + m_ji[ji_idx])
        # Apply the angle dense layers
        angle_feats = angle_prod
        for angle_dense_layer in self.angle_dense_layers:
            angle_feats = angle_dense_layer(angle_feats)

        return edge_feats, angle_feats


class ForceDime(nn.Module):

    """ForceDime Implementation
    Attributes:
        radial_basis (nff.nn.RadialBasis): radial basis layers for
            distances.
        spherical_basis (nff.nn.SphericalBasis): spherical basis for
            both distances and angles.
        embedding_block (nff.nn.EmbeddingBlock): block to convert
            atomic numbers into embedding vectors and concatenate
            embeddings and distances to make message embeddings.
        interaction_blocks (nn.ModuleList[nff.nn.InteractionBlock]):
            blocks for aggregating distance and angle information
            from neighboring atoms.
        output_blocks (nn.ModuleDict): Module  dictionary. Each
            key of the dictionary corresponds to a different property
            prediction, and its value is of type nn.ModuleList[nff.nn.
            OutputBlock]. These output blocks aggregate information
            at each interaction block and add it to the final result.
        out_keys (list): list of properties to be predicted by the
            network.
        grad_keys (list): list of properties for which we want the
            gradient to be computed.

    """

    def __init__(self, n_rbf, cutoff, 
                        envelope_p, l_spher, 
                        n_spher, embed_dim, 
                        activation, 
                        n_bilinear, 
                        n_convolutions,
                        **kwargs):

        super().__init__()

        self.radial_basis = RadialBasis(
            n_rbf=n_rbf,
            cutoff=cutoff,
            envelope_p=envelope_p)

        self.spherical_basis = SphericalBasis(
            n_spher=n_spher,
            l_spher=l_spher,
            cutoff=cutoff,
            envelope_p=envelope_p)

        self.embedding_block = EmbeddingBlock(
            n_rbf=n_rbf,
            embed_dim=embed_dim,
            activation=activation)

        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(embed_dim=embed_dim,
                             n_rbf=n_rbf,
                             activation=activation,
                             n_spher=n_spher,
                             l_spher=l_spher,
                             n_bilinear=n_bilinear)
            for _ in range(n_convolutions)
        ])
        
        self.readout_blocks = nn.ModuleList([
            ReadoutBlock(embed_dim=embed_dim,
                         n_rbf=n_rbf,
                         n_sbf=l_spher*n_spher,
                         activation=activation)
            for _ in range(n_convolutions + 1)
        ])

    def get_prelims(self, batch):
        
        nbr_list = batch["nbr_list"]
        angle_list = batch["angle_list"]
        nxyz = batch["nxyz"]
        num_atoms = batch["num_atoms"].sum()

        xyz = nxyz[:, 1:]
        z = nxyz[:, 0].long()
        xyz.requires_grad = False

        ji_idx = batch["ji_idx"]
        kj_idx = batch["kj_idx"]

        # compute distances
        d = torch.norm(xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]],
                       dim=-1).reshape(-1, 1)

        # compute angles
        alpha = compute_angle(xyz, angle_list)

        # put the distances in the radial basis
        e_rbf = self.radial_basis(d)

        # put the distances and angles in the spherical basis
        a_sbf = self.spherical_basis(d, alpha, kj_idx)

        return (xyz, e_rbf, a_sbf, nbr_list, angle_list, num_atoms,
                z, kj_idx, ji_idx)

    def atomwise(self, e_rbf, a_sbf, nbr_list, angle_list,
                         num_atoms, z, kj_idx, ji_idx):

        # embed edge vectors
        m_ji = self.embedding_block(e_rbf=e_rbf,
                                    z=z,
                                    nbr_list=nbr_list)
        edge_feats, angle_feats = self.readout_blocks[0](m_ji=m_ji,
                                                        e_rbf=e_rbf,
                                                        a_sbf=a_sbf,
                                                        nbr_list=nbr_list,
                                                        kj_idx=kj_idx, 
                                                        ji_idx=ji_idx)
        
        # cycle through the interaction blocks
        for i, int_block in enumerate(self.interaction_blocks):

            # update the edge vector
            m_ji = m_ji + int_block(m_ji=m_ji,
                             e_rbf=e_rbf,
                             a_sbf=a_sbf,
                             kj_idx=kj_idx,
                             ji_idx=ji_idx)
            edge_feats_update, angle_feats_update = self.readout_blocks[i+1](m_ji=m_ji,
                                                                            e_rbf=e_rbf,
                                                                            a_sbf=a_sbf,
                                                                            nbr_list=nbr_list,
                                                                            kj_idx=kj_idx, 
                                                                            ji_idx=ji_idx)
            edge_feats += edge_feats_update
            angle_feats += angle_feats_update
            
        return edge_feats, angle_feats

    def forward(self, batch):
        
        xyz, e_rbf, a_sbf, nbr_list, angle_list, num_atoms, z, kj_idx, ji_idx = self.get_prelims(batch)
        
        edge_feats, angle_feats = self.atomwise(e_rbf, a_sbf, nbr_list, angle_list, num_atoms, z, kj_idx, ji_idx)

        # f_edge = self.edgereadout( m_ji ) * dis_vec
        
        # dis_vec = xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]]
    
        # f_atom = scatter_add(f_edge, nbr_list[:,0], dim=0, dim_size=num_atoms) - \
        #     scatter_add(f_edge, nbr_list[:,1], dim=0, dim_size=num_atoms)
        
        # results = dict()
        # results['energy_grad'] = f_atom
        
        # TODO: finish invariant edge and angle feats, should implement adjoints now
        return edge_feats, angle_feats
