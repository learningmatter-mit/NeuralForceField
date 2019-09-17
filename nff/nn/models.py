import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

from nff.nn.layers import Dense, GaussianSmearing
from nff.nn.modules import GraphDis, SchNetConv, BondEnergyModule, SchNetEdgeUpdate, NodeMultiTaskReadOut
from nff.nn.activations import shifted_softplus
from nff.nn.graphop import batch_and_sum, get_atoms_inside_cell

DEFAULT_READOUT = {
    'energy': [
        {'name': 'linear', 'param' : { 'in_features': n_atom_basis, 'out_features': int(n_atom_basis / 2)}},
        {'name': 'shifted_softplus', 'param': {}},
        {'name': 'linear', 'param' : { 'in_features': int(n_atom_basis / 2), 'out_features': 1}}
    ]
}

class SchNet(nn.Module):

    """SchNet implementation with continous filter.
    
    Attributes:
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis

        atomwise1 (Dense): dense layer 1 to compute energy
        atomwise2 (Dense): dense layer 2 to compute energy
        convolutions (torch.nn.ModuleList): include all the convolutions
        graph_dis (Graphdis): graph distance module to convert xyz inputs
            into distance matrix
        prop_dics (dict): A dictionary of the form {name: prop_dic}, where name is the
            property name and prop_dic is a dictionary for that property.
        module_dict (ModuleDict): a dictionary of modules. Each entry has the form
            {name: mod_list}, where name is the name of a property object and mod_list
            is a ModuleList of layers to predict that property.
    """
    
    def __init__(self, modelparams):
        """Constructs a SchNet model.
        
        Args:
            modelparams (TYPE): Description
        """

        super().__init__()

        n_atom_basis = modelparams['n_atom_basis']
        n_filters = modelparams['n_filters']
        n_gaussians = modelparams['n_gaussians']
        n_convolutions = modelparams['n_convolutions']
        cutoff = modelparams['cutoff']
        trainable_gauss = modelparams.get('trainable_gauss', False)
        box_size = modelparams.get('box_size', None)

        # default predict var
        readoutdict = modelparams.get('readoutdict', DEFAULT_READOUT)
        post_readout =  modelparams.get('post_readout', None)

        self.graph_dis = GraphDis(Fr=1,
                                  Fe=1,
                                  cutoff=cutoff,
                                  box_size=box_size)

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        self.convolutions = nn.ModuleList([
            SchNetConv(n_atom_basis=n_atom_basis,
                             n_filters=n_filters,
                             n_gaussians=n_gaussians, 
                             cutoff=cutoff,
                             trainable_gauss=trainable_gauss)
            for _ in range(n_convolutions)
        ])

        # ReadOut
        self.atomwisereadout = NodeMultiTaskReadOut(multitaskdict=readoutdict, post_readout=post_readout)        
        self.device = None

    def forward(self, batch, other_results=False):

        """Summary
        
        Args:
            batch (dict): dictionary of props
        
        Returns:
            dict: dionary of results 
        """
        r = batch['nxyz'][:, 0]
        xyz = batch['nxyz'][:, 1:4]
        N = batch['num_atoms'].to(torch.long)
        a = batch.get('nbr_list', None)
        pbc = batch.get('pbc', None)

        xyz.requires_grad = True

        # a is None means non-batched case
        if a is None:
            assert len(set(N)) == 1 # all the graphs should correspond to the same molecule
            N_atom = N[0]
            e, a = self.graph_dis(xyz=xyz.reshape(-1, N_atom, 3))

            if pbc is None:
                pbc = torch.LongTensor(range(r.shape[0]))
            pbc = pbc.to(self.device)

        # batched case
        else:
            # calculating the distances
            e = (xyz[a[:, 0]] - xyz[a[:, 1]]).pow(2).sum(1).sqrt()[:, None]

        a = a.to(self.device)

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()[pbc]

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr
            r = r[pbc]

        # remove image atoms outside the unit cell
        r, N = get_atoms_inside_cell(r, N, pbc)

        r = self.atomwisereadout(r)

        results = batch_and_sum(r, N, list(batch.keys()), xyz)
        
        return results 

