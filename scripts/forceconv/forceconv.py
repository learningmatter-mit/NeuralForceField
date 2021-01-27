import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import torch 
from torch.nn import Embedding
from torch.nn import Sequential 
from nff.nn.layers import GaussianSmearing, Dense
from nff.nn.activations import shifted_softplus
from nff.nn.graphconv import MessagePassingModule
from nff.utils.scatter import scatter_add
from torch.nn import ModuleDict
from torch import nn


from nff.nn.modules.dimenet import (EmbeddingBlock, InteractionBlock,
                                    OutputBlock)
from nff.nn.layers import DimeNetRadialBasis as RadialBasis
from nff.nn.layers import DimeNetSphericalBasis as SphericalBasis
from nff.utils.scatter import compute_grad

from nff.nn.models import DimeNet
from nff.nn.models.dimenet import compute_angle


nlr_dict =  {
    'ReLU': nn.ReLU(), 
    'ELU': nn.ELU(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(),
    'ReLU6':nn.ReLU6(),
    'SELU': nn.SELU(),
    'CELU': nn.CELU(),
    'Tanhshrink': nn.Tanhshrink(),
    'ssf': shifted_softplus()
}

class EdgeConv(MessagePassingModule):

    """The convolution layer with filter.
    Attributes:
        moduledict (TYPE): Description
    """

    def __init__(
        self,
        n_atom_basis,
        n_edge_basis,
        n_filters,
        atom_filter_depth,
        edge_filter_depth,
        atom_update_depth,
        edge_update_depth,

    ):
        super(EdgeConv, self).__init__()

        # construct edge filter networks 
        edge_filter = [Dense(in_features=n_edge_basis, out_features=n_edge_basis), shifted_softplus()]
        for i in range(edge_filter_depth):
            edge_filter.append( Dense(in_features=n_edge_basis, out_features=n_edge_basis))
            edge_filter.append(shifted_softplus())

        edge_filter.append(Dense(in_features=n_edge_basis, out_features=n_filters))

        # construct atom filter networks
        atom_filter = [Dense(in_features=n_atom_basis, out_features=n_atom_basis), shifted_softplus()]
        for i in range(atom_filter_depth):
            atom_filter.append(Dense(in_features=n_atom_basis, out_features=n_atom_basis))
            atom_filter.append(shifted_softplus())
        atom_filter.append(Dense(in_features=n_atom_basis, out_features=n_filters))

        # construct edge update networks
        edge_update = [Dense(in_features=n_filters, out_features=n_filters), shifted_softplus()]
        for i in range(edge_update_depth):
            edge_update.append(Dense(in_features=n_filters, out_features=n_filters))
            edge_update.append(shifted_softplus())
        edge_update.append(Dense(in_features=n_filters, out_features=n_edge_basis))

        # construct atom update networks
        atom_update = [Dense(in_features=n_filters, out_features=n_filters), shifted_softplus()]
        for i in range(atom_update_depth):
            atom_update.append(Dense(in_features=n_filters, out_features=n_filters))
            atom_update.append(shifted_softplus())
        atom_update.append(Dense(in_features=n_filters, out_features=n_atom_basis))

        self.moduledict = ModuleDict(
            {
                "edge_filter": Sequential(*edge_filter),
                "atom_filter": Sequential(*atom_filter),
                "atom_update_function": Sequential(*atom_update),
                "edge_update_function": Sequential(*edge_update),
            }
        )

    def message(self, r, e, a):
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
        e = self.moduledict["edge_filter"](e)
        # convection: update
        r = self.moduledict["atom_filter"](r)
        # combine node and edge info
        message = r[a[:, 0]] * e, r[a[:, 1]] * e
        return message

    def update(self, r):
        return self.moduledict["update_function"](r)
    
    def forward(self, r, e, a):

        graph_size = r.shape[0]

        rij, rji = self.message(r, e, a)
        dr = self.aggregate(rij, a[:, 1], graph_size)
        dr += self.aggregate(rji, a[:, 0], graph_size)
        
        dr = self.moduledict['atom_update_function'](dr)
        de = self.moduledict['edge_update_function'](rij + rji)
        
        return dr, de
    
    
class ForceConvolve(torch.nn.Module):
    
    def __init__(self, n_convolutions, n_edge_basis, n_atom_basis, n_filters, n_gaussians, cutoff,
                    edge_filter_depth, atom_filter_depth, edge_update_depth, atom_update_depth, **kwargs):
        torch.nn.Module.__init__(self)
        # atom transform
        self.atom_filter = Embedding(100, n_atom_basis)
        
        # distance transform
        self.smear = GaussianSmearing(start=0.0, stop=cutoff, n_gaussians=n_gaussians)
        
        self.edgefilter = Sequential(
            Dense(in_features=n_gaussians, out_features=n_edge_basis),
            shifted_softplus(),
            Dense(in_features=n_edge_basis, out_features=n_edge_basis))
        
        # convolutions 
        self.conv = torch.nn.ModuleList(
            [ EdgeConv(n_atom_basis=n_atom_basis,
                        n_edge_basis=n_edge_basis, 
                        n_filters=n_filters,
                        edge_filter_depth=edge_filter_depth,
                        atom_filter_depth=atom_filter_depth,
                        edge_update_depth=edge_update_depth,
                        atom_update_depth=atom_update_depth)
                for _ in range(n_convolutions)
            ]
        )
        
        # edge readout 
        self.edgereadout = Sequential(
            Dense(in_features=n_edge_basis, out_features=n_edge_basis),
            shifted_softplus(),
            Dense(in_features=n_edge_basis, out_features=1))
        
        
    def forward(self, batch, xyz=None):
        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]
            xyz.requires_grad = False
        
        r = batch["nxyz"][:, 0]
        a = batch["nbr_list"]
        
        graph_size = r.shape[0]
        
        # edge feature 
        dis_vec = xyz[a[:, 0]] - xyz[a[:, 1]]
        dis = dis_vec.pow(2).sum(1).sqrt()[:, None]
        
        xyz_adjoint = dis_vec / dis
        
        e = self.smear(dis)
        e = self.edgefilter(e)
        
        # node feature
        r = self.atom_filter(r.long()).squeeze()
        
        for i, conv in enumerate(self.conv):
            dr, de = conv(r, e, a)
            r = r + dr 
            e = e + de
        
        f_edge = self.edgereadout(e) * xyz_adjoint
        
        f_atom = scatter_add(f_edge, a[:,0], dim=0, dim_size=graph_size) - \
            scatter_add(f_edge, a[:,1], dim=0, dim_size=graph_size)
        
        results = dict()
        results['energy_grad'] = f_atom
        
        return results



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
        
        
        self.edgereadout = Sequential(Dense(in_features=embed_dim, 
                                      out_features=embed_dim),
                                      shifted_softplus(),
                                      Dense(in_features=embed_dim, out_features=1))

    def get_prelims(self, batch):
        
        nbr_list = batch["nbr_list"]
        angle_list = batch["angle_list"]
        nxyz = batch["nxyz"]
        num_atoms = batch["num_atoms"].sum()

        xyz = nxyz[:, 1:]
        z = nxyz[:, 0].long()
        xyz.requires_grad = True

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
        
        # cycle through the interaction blocks
        for i, int_block in enumerate(self.interaction_blocks):

            # update the edge vector
            m_ji = m_ji + int_block(m_ji=m_ji,
                             e_rbf=e_rbf,
                             a_sbf=a_sbf,
                             kj_idx=kj_idx,
                             ji_idx=ji_idx)
            
        return m_ji

    def forward(self, batch):
        
        xyz, e_rbf, a_sbf, nbr_list, angle_list, num_atoms, z, kj_idx, ji_idx = self.get_prelims(batch)
        
        m_ji = self.atomwise(e_rbf, a_sbf, nbr_list, angle_list, num_atoms, z, kj_idx, ji_idx)
        
        dis_vec = xyz[nbr_list[:, 0]] - xyz[nbr_list[:, 1]]
        
        f_edge = self.edgereadout( m_ji ) * dis_vec
    
        f_atom = scatter_add(f_edge, nbr_list[:,0], dim=0, dim_size=num_atoms) - \
            scatter_add(f_edge, nbr_list[:,1], dim=0, dim_size=num_atoms)
        
        results = dict()
        results['energy_grad'] = f_atom
        
        return results
