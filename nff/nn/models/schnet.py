from torch import nn

from nff.nn.layers import DEFAULT_DROPOUT_RATE
from nff.nn.modules import (
    SchNetConv,
    NodeMultiTaskReadOut,
    get_rij,
    add_stress
)


from nff.nn.modules.diabat import DiabaticReadout
from nff.nn.graphop import batch_and_sum
from nff.nn.utils import get_default_readout

from nff.utils.scatter import scatter_add


class SchNet(nn.Module):

    """SchNet implementation with continous filter.

    Attributes:
        atom_embed (torch.nn.Embedding): Convert atomic number into an
            embedding vector of size n_atom_basis
        convolutions (torch.nn.Module): convolution layers applied to the graph
        atomwisereadout (torch.nn.Module): fully connected layers applied to the graph
            to get the results of interest
        device (int): GPU being used.


    """

    def __init__(self, modelparams):
        """Constructs a SchNet model.

        Args:
            modelparams (TYPE): Description

        Example:

            n_atom_basis = 256

            readoutdict = {
                                "energy_0": [{'name': 'linear', 'param' : { 'in_features': n_atom_basis,
                                                                          'out_features': int(n_atom_basis / 2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int(n_atom_basis / 2),
                                                                          'out_features': 1}}],
                                "energy_1": [{'name': 'linear', 'param' : { 'in_features': n_atom_basis,
                                                                          'out_features': int(n_atom_basis / 2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int(n_atom_basis / 2),
                                                                          'out_features': 1}}]
                            }


            modelparams = {
                'n_atom_basis': n_atom_basis,
                'n_filters': 256,
                'n_gaussians': 32,
                'n_convolutions': 4,
                'cutoff': 5.0,
                'trainable_gauss': True,
                'readoutdict': readoutdict,    
                'dropout_rate': 0.2
            }

            model = SchNet(modelparams)

        """

        nn.Module.__init__(self)

        n_atom_basis = modelparams["n_atom_basis"]
        n_filters = modelparams["n_filters"]
        n_gaussians = modelparams["n_gaussians"]
        n_convolutions = modelparams["n_convolutions"]
        cutoff = modelparams["cutoff"]
        trainable_gauss = modelparams.get("trainable_gauss", False)
        dropout_rate = modelparams.get("dropout_rate", DEFAULT_DROPOUT_RATE)

        self.excl_vol = modelparams.get("excl_vol", False)
        if self.excl_vol:
            self.power = modelparams["V_ex_power"]
            self.sigma = modelparams["V_ex_sigma"]

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

        readoutdict = modelparams.get(
            "readoutdict", get_default_readout(n_atom_basis))
        post_readout = modelparams.get("post_readout", None)

        # convolutions
        self.convolutions = nn.ModuleList(
            [
                SchNetConv(
                    n_atom_basis=n_atom_basis,
                    n_filters=n_filters,
                    n_gaussians=n_gaussians,
                    cutoff=cutoff,
                    trainable_gauss=trainable_gauss,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_convolutions)
            ]
        )

        # ReadOut
        self.atomwisereadout = NodeMultiTaskReadOut(
            multitaskdict=readoutdict, post_readout=post_readout
        )
        self.device = None
        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        gauss_centers = (self.convolutions[0].moduledict
                         ['message_edge_filter'][0].offsets)
        self.cutoff = gauss_centers[-1] - gauss_centers[0]

    def convolve(self,
                 batch,
                 xyz=None):
        """

        Apply the convolutional layers to the batch.

        Args:
            batch (dict): dictionary of props

        Returns:
            r: new feature vector after the convolutions
            N: list of the number of atoms for each molecule in the batch
            xyz: xyz (with a "requires_grad") for the batch
        """

        # Note: we've given the option to input xyz from another source.
        # E.g. if you already created an xyz  and set requires_grad=True,
        # you don't want to make a whole new one.

        if xyz is None:
            xyz = batch["nxyz"][:, 1:4]
            if xyz.requires_grad == False:
                xyz.requires_grad = True

        r = batch["nxyz"][:, 0]
        N = batch["num_atoms"].reshape(-1).tolist()
        a = batch["nbr_list"]

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, a = get_rij(xyz=xyz,
                          batch=batch,
                          nbrs=a,
                          cutoff=self.cutoff)
        dist = r_ij.pow(2).sum(1).sqrt()
        e = dist[:, None]

        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        return r, N, xyz, r_ij, a

    def V_ex(self, r_ij, nbr_list, xyz):

        dist = (r_ij).pow(2).sum(1).sqrt()
        potential = ((dist.reciprocal() * self.sigma).pow(self.power))

        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]

    def forward(self,
                batch,
                xyz=None,
                requires_stress=False,
                **kwargs):
        """Summary
        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates

        Returns:
            dict: dictionary of results

        """

        r, N, xyz, r_ij, a = self.convolve(batch, xyz)
        r = self.atomwisereadout(r)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, a, xyz)
            r['energy'] += r_ex

        results = batch_and_sum(r, N, list(batch.keys()), xyz)
        if requires_stress:
            results = add_stress(batch=batch,
                                 all_results=results,
                                 nbrs=a,
                                 r_ij=r_ij)

        return results


class SchNetDiabat(SchNet):
    def __init__(self, modelparams):

        super().__init__(modelparams)

        self.diabatic_readout = DiabaticReadout(
            diabat_keys=modelparams["diabat_keys"],
            grad_keys=modelparams["grad_keys"],
            energy_keys=modelparams["output_keys"])

    def forward(self,
                batch,
                xyz=None,
                add_nacv=False,
                add_grad=True,
                add_gap=True,
                extra_grads=None,
                try_speedup=False,
                **kwargs):

        r, N, xyz = self.convolve(batch, xyz)
        output = self.atomwisereadout(r)
        results = self.diabatic_readout(batch=batch,
                                        output=output,
                                        xyz=xyz,
                                        add_nacv=add_nacv,
                                        add_grad=add_grad,
                                        add_gap=add_gap,
                                        extra_grads=extra_grads,
                                        try_speedup=try_speedup)

        return results
