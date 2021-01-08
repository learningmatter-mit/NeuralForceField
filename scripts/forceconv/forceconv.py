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
    ):
        super(EdgeConv, self).__init__()
        self.moduledict = ModuleDict(
            {
                "edge_filter": Sequential(
                    Dense(
                        in_features=n_edge_basis,
                        out_features=n_edge_basis,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_edge_basis,
                        out_features=n_filters,
                    )
                ),
                "atom_filter": Sequential(
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_filters,
                    ),
                ),
                "atom_update_function": Sequential(
                    Dense(
                        in_features=n_filters,
                        out_features=n_atom_basis,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_atom_basis,
                        out_features=n_atom_basis,
                    ),
                
                ),
                "edge_update_function": Sequential(
                    Dense(
                        in_features=n_filters,
                        out_features=n_edge_basis,
                    ),
                    shifted_softplus(),
                    Dense(
                        in_features=n_edge_basis,
                        out_features=n_edge_basis,
                    )
                )
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
    
    def __init__(self, n_convolutions, n_edge_basis, n_atom_basis, n_filters, n_gaussians, cutoff):
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
            [ EdgeConv(n_atom_basis=n_atom_basis, n_edge_basis=n_edge_basis, n_filters=n_filters)
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
            xyz.requires_grad = True
        
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