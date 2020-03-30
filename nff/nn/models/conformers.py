import torch
import torch.nn as nn

from nff.nn.layers import DEFAULT_DROPOUT_RATE
from nff.nn.modules import (
    SchNetConv,
    NodeMultiTaskReadOut,
)
from nff.nn.graphop import conf_pool
from nff.nn.utils import construct_sequential

"""
Model that uses a representation of a molecule in terms of different 3D
conformers to predict properties.
"""


class WeightedConformers(nn.Module):

    def __init__(self, modelparams):
        """Constructs a SchNet-Like model using a conformer representation.

        Args:
            modelparams (dict): dictionary of parameters for model. All
                are the same as in SchNet, except for  `mol_fp_layers`,
                which describes how to convert atomic fingerprints into
                a single molecular fingerprint.

        Example:

            n_atom_basis = 256
            mol_basis = 512

            # all the atomic fingerprints get added together, then go through the network created
            # by `mol_fp_layers` to turn into a molecular fingerprint
            mol_fp_layers = [{'name': 'linear', 'param' : { 'in_features': n_atom_basis,
                                                                          'out_features': int((n_atom_basis + mol_basis)/2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int((n_atom_basis + mol_basis)/2),
                                                                          'out_features': mol_basis}}]


            readoutdict = {
                                "covid": [{'name': 'linear', 'param' : { 'in_features': mol_basis,
                                                                          'out_features': int(mol_basis / 2)}},
                                           {'name': 'shifted_softplus', 'param': {}},
                                           {'name': 'linear', 'param' : { 'in_features': int(mol_basis / 2),
                                                                          'out_features': 1}},
                                           {'name': 'sigmoid', 'param': {}}],
                            }

            # dictionary to tell you what to do with the Boltzmann factors
            # ex. 1:

            boltzmann_dict = {"type": "multiply"}

            # ex. 2
            boltzmann_layers = [{'name': 'linear', 'param': {'in_features': mol_basis + 1,
                                                           'out_features': mol_basis}},
                                {'name': 'shifted_softplus', 'param': {}},
                                {'name': 'linear', 'param': {'in_features': mol_basis,
                                                           'out_features': mol_basis}}]
            boltzmann_dict = {"type": "layers", "layers": boltzmann_layers}


            modelparams = {
                'n_atom_basis': n_atom_basis,
                'n_filters': 256,
                'n_gaussians': 32,
                'n_convolutions': 4,
                'cutoff': 5.0,
                'trainable_gauss': True,
                'readoutdict': readoutdict,    
                'mol_fp_layers': mol_fp_layers,
                'boltzmann_dict': boltzmann_dict
                'dropout_rate': 0.2
            }

            model = WeightedConformers(modelparams)

        """

        nn.Module.__init__(self)

        n_atom_basis = modelparams["n_atom_basis"]
        n_filters = modelparams["n_filters"]
        n_gaussians = modelparams["n_gaussians"]
        n_convolutions = modelparams["n_convolutions"]
        cutoff = modelparams["cutoff"]
        trainable_gauss = modelparams.get("trainable_gauss", False)
        dropout_rate = modelparams.get("dropout_rate", DEFAULT_DROPOUT_RATE)

        self.atom_embed = nn.Embedding(100, n_atom_basis, padding_idx=0)

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

        mol_fp_layers = modelparams["mol_fp_layers"]
        readoutdict = modelparams["readoutdict"]
        boltzmann_dict = modelparams["boltzmann_dict"]


        # the nn that converts atomic finerprints to a molecular fp
        self.mol_fp_nn = construct_sequential(mol_fp_layers)

        # create a module that lets a molecular fp interact with the
        # conformer's boltzmann weight to give a final molecular fp
        self.boltz_nn = self.make_boltz_nn(boltzmann_dict)

        # the readout acts on this final molceular fp
        self.readout = NodeMultiTaskReadOut(multitaskdict=readoutdict)


    def make_boltz_nn(self, boltzmann_dict):
        if boltzmann_dict["type"] == "multiply":
            return
        layers = boltzmann_dict["layers"]
        network = construct_sequential(layers)
        return network


    def convolve(self, batch, xyz=None):
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
            xyz.requires_grad = True

        r = batch["nxyz"][:, 0]
        mol_sizes = batch["mol_size"].reshape(-1).tolist()
        N = batch["num_atoms"].reshape(-1).tolist()
        num_confs = (torch.tensor(N) / torch.tensor(mol_sizes)).tolist()

        a = batch["nbr_list"]

        # offsets take care of periodic boundary conditions
        offsets = batch.get("offsets", 0)

        e = (xyz[a[:, 0]] - xyz[a[:, 1]] -
             offsets).pow(2).sum(1).sqrt()[:, None]

        # ensuring image atoms have the same vectors of their corresponding
        # atom inside the unit cell
        r = self.atom_embed(r.long()).squeeze()

        # update function includes periodic boundary conditions
        for i, conv in enumerate(self.convolutions):
            dr = conv(r=r, e=e, a=a)
            r = r + dr

        # split the fingerprints by species
        fps_by_smiles = torch.split(r, N)
        # split the boltzmann weights by species
        boltzmann_weights = torch.split(batch["weights"], num_confs)
        # get the morgan fp per species if it exists
        if "morgan" in batch.keys():
            num_mols = len(fps_by_smiles)
            morgan_len = len(batch["morgan"]) // num_mols
            splits = [morgan_len] * num_mols
            morgan = torch.split(batch["morgan"], splits)
        else:
            morgan = None
        # return everything in a dictionary
        outputs = dict(r=r,
                       N=N,
                       xyz=xyz,
                       fps_by_smiles=fps_by_smiles,
                       boltzmann_weights=boltzmann_weights,
                       mol_sizes=mol_sizes,
                       morgan=morgan)

        return outputs

    def forward(self, batch, xyz=None):
        """

        Use the outputs of the convolutions to make a prediction.
        Here, the atomic fingerprints for each geometry get converted
        into a molecular fingerprint. Then, the molecular
        fingerprints for the different conformers of a given species
        get multiplied by the Boltzmann weights of those conformers and
        added together to make a final fingerprint for the species.
        Two fully-connected layers act on this final fingerprint to make
        a prediction.

        Args:
            batch (dict): dictionary of props
            xyz (torch.tensor): (optional) coordinates

        Returns:
            dict: dictionary of results

        """

        outputs = self.convolve(batch, xyz)

        fps_by_smiles = outputs["fps_by_smiles"]
        batched_weights = outputs["boltzmann_weights"]
        mol_sizes = outputs["mol_sizes"]
        morgan_fps = outputs["morgan"]

        conf_fps = []
        # go through each species
        for i in range(len(fps_by_smiles)):
            boltzmann_weights = batched_weights[i]
            smiles_fp = fps_by_smiles[i]
            mol_size = mol_sizes[i]
            morgan = morgan_fps[i]

            # pool the atomic fingerprints as described above
            conf_fp = conf_pool(smiles_fp=smiles_fp,
                                mol_size=mol_size,
                                boltzmann_weights=boltzmann_weights,
                                mol_fp_nn=self.mol_fp_nn,
                                boltz_nn=self.boltz_nn,
                                morgan_fp=morgan)

            conf_fps.append(conf_fp)

        conf_fps = torch.stack(conf_fps)
        results = self.readout(conf_fps)

        return results
