import torch
from torch import nn
from nff.utils.scatter import compute_grad
from nff.utils.tools import make_directed, make_undirected
from nff.nn.modules.spooky import (DEFAULT_DROPOUT, DEFAULT_ACTIVATION,
                                   DEFAULT_MAX_Z, CombinedEmbedding,
                                   InteractionBlock, AtomwiseReadout,
                                   Electrostatics, NuclearRepulsion,
                                   get_dipole)


class SpookyNet(nn.Module):
    def __init__(self,
                 modelparams):
        super().__init__()

        output_keys = modelparams['output_keys']
        grad_keys = modelparams['grad_keys']
        feat_dim = modelparams['feat_dim']
        r_cut = modelparams['r_cut']
        gamma = modelparams['gamma']
        bern_k = modelparams['bern_k']
        num_conv = modelparams['num_conv']
        add_disp = modelparams.get('add_disp', False)
        add_nuc_keys = modelparams.get('add_nuc_keys')
        dropout = modelparams.get('dropout', DEFAULT_DROPOUT)
        activation = modelparams.get('activation', DEFAULT_ACTIVATION)
        max_z = modelparams.get('max_z', DEFAULT_MAX_Z)

        if add_disp:
            raise NotImplementedError("Dispersion not implemented")

        self.output_keys = output_keys
        self.grad_keys = grad_keys
        self.add_nuc_keys = (add_nuc_keys if (add_nuc_keys is not None)
                             else output_keys)
        self.embedding = CombinedEmbedding(feat_dim,
                                           activation=activation,
                                           max_z=max_z)
        self.interactions = nn.ModuleList([
            InteractionBlock(feat_dim=feat_dim,
                             r_cut=r_cut,
                             gamma=gamma,
                             bern_k=bern_k,
                             activation=activation,
                             dropout=dropout,
                             max_z=max_z)
            for _ in range(num_conv)
        ])
        self.atomwise_readout = nn.ModuleDict(
            {
                key: AtomwiseReadout(feat_dim=feat_dim)
                for key in output_keys
            }
        )
        self.electrostatics = nn.ModuleDict(
            {
                key: Electrostatics(feat_dim=feat_dim,
                                    r_cut=r_cut,
                                    max_z=max_z)
                for key in output_keys
            }
        )
        self.nuc_repulsion = NuclearRepulsion(r_cut=r_cut)

    def fwd(self,
            batch,
            xyz=None,
            grad_keys=None):

        nxyz = batch['nxyz']
        nbrs, _ = make_directed(batch['nbr_list'])
        mol_nbrs = make_undirected(batch['mol_nbrs'])

        z = nxyz[:, 0].long()
        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        charge = batch['charge']
        spin = batch['spin']
        num_atoms = batch['num_atoms']

        x = self.embedding(charge=charge,
                           spin=spin,
                           z=z,
                           num_atoms=num_atoms)

        f = torch.zeros_like(x)
        for i, interaction in enumerate(self.interactions):
            x, y_t = interaction(x=x,
                                 xyz=xyz,
                                 nbrs=nbrs,
                                 num_atoms=num_atoms)

            f = f + y_t

        results = {}
        for key in self.output_keys:

            atomwise_readout = self.atomwise_readout[key]
            electrostatics = self.electrostatics[key]

            learned_e = atomwise_readout(z=z,
                                         f=f,
                                         num_atoms=num_atoms)
            elec_e, q = electrostatics(f=f,
                                       z=z,
                                       xyz=xyz,
                                       total_charge=charge,
                                       num_atoms=num_atoms,
                                       mol_nbrs=mol_nbrs)

            total_e = learned_e + elec_e

            if key in self.add_nuc_keys:
                nuc_e = self.nuc_repulsion(xyz=xyz,
                                           z=z,
                                           nbrs=nbrs,
                                           num_atoms=num_atoms)
                total_e += nuc_e

            dipole = get_dipole(xyz=xyz,
                                q=q,
                                num_atoms=num_atoms)

            suffix = "_" + key.split("_")[-1]
            if not any([i.isdigit() for i in suffix]):
                suffix = ""

            results.update({key: total_e,
                            f"q{suffix}": q,
                            f"dipole{suffix}": dipole})

        if grad_keys is None:
            grad_keys = self.grad_keys

        for key in grad_keys:
            base_key = key.replace("_grad", "")
            grad = compute_grad(inputs=xyz,
                                output=results[base_key])
            results[key] = grad

        return results

    def forward(self, *args, **kwargs):
        try:
            return self.fwd(*args, **kwargs)
        except Exception as e:
            print(e)
            import pdb
            pdb.post_mortem()
