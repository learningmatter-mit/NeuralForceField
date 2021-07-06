import torch
from torch import nn
from nff.utils.scatter import compute_grad
from nff.utils.tools import make_directed, make_undirected
from nff.nn.modules.spooky_fast import (DEFAULT_DROPOUT, DEFAULT_ACTIVATION,
                                        DEFAULT_MAX_Z, DEFAULT_RES_LAYERS,
                                        CombinedEmbedding, InteractionBlock,
                                        AtomwiseReadout, Electrostatics,
                                        NuclearRepulsion, get_dipole)


def default(val, def_val):
    out = val if (val is not None) else def_val
    return out


def parse_optional(modelparams):
    dropout = default(modelparams.get('dropout'),
                      DEFAULT_DROPOUT)
    activation = default(modelparams.get('activation'),
                         DEFAULT_ACTIVATION)
    max_z = default(modelparams.get('max_z'), DEFAULT_MAX_Z)
    residual_layers = default(modelparams.get('residual_layers'),
                              DEFAULT_RES_LAYERS)
    return dropout, activation, max_z, residual_layers


def parse_add_ons(modelparams):
    add_nuc_keys = default(modelparams.get('add_nuc_keys'),
                           modelparams['output_keys'])
    add_elec_keys = default(modelparams.get('add_elec_keys'),
                            modelparams['output_keys'])
    add_disp_keys = default(modelparams.get('add_disp_keys'),
                            [])

    return add_nuc_keys, add_elec_keys, add_disp_keys


class SpookyNet(nn.Module):
    def __init__(self,
                 modelparams):

        super().__init__()

        feat_dim = modelparams['feat_dim']
        r_cut = modelparams['r_cut']
        optional = parse_optional(modelparams)
        dropout, activation, max_z, residual_layers = optional
        add_ons = parse_add_ons(modelparams)
        add_nuc_keys, add_elec_keys, add_disp_keys = add_ons

        self.output_keys = modelparams['output_keys']
        self.grad_keys = modelparams['grad_keys']
        self.embedding = CombinedEmbedding(feat_dim=feat_dim,
                                           activation=activation,
                                           max_z=max_z,
                                           residual_layers=residual_layers)
        self.interactions = nn.ModuleList([
            InteractionBlock(feat_dim=feat_dim,
                             r_cut=r_cut,
                             gamma=modelparams['gamma'],
                             bern_k=modelparams['bern_k'],
                             activation=activation,
                             dropout=dropout,
                             max_z=max_z,
                             residual_layers=residual_layers)
            for _ in range(modelparams['num_conv'])
        ])
        self.atomwise_readout = nn.ModuleDict({
            key: AtomwiseReadout(feat_dim=feat_dim)
            for key in self.output_keys
        })

        self.electrostatics = nn.ModuleDict({
            key: Electrostatics(feat_dim=feat_dim,
                                r_cut=r_cut,
                                max_z=max_z)
            for key in add_elec_keys
        })

        self.nuc_repulsion = nn.ModuleDict({
            key: NuclearRepulsion(r_cut=r_cut)
            for key in add_nuc_keys
        })

        if add_disp_keys:
            raise NotImplementedError("Dispersion not implemented")

    def get_results(self,
                    z,
                    f,
                    num_atoms,
                    xyz,
                    charge,
                    mol_nbrs,
                    nbrs):

        results = {}
        for key in self.output_keys:
            atomwise_readout = self.atomwise_readout[key]
            energy = atomwise_readout(z=z,
                                      f=f,
                                      num_atoms=num_atoms)

            if key in self.electrostatics:
                electrostatics = self.electrostatics[key]
                elec_e, q = electrostatics(f=f,
                                           z=z,
                                           xyz=xyz,
                                           total_charge=charge,
                                           num_atoms=num_atoms,
                                           mol_nbrs=mol_nbrs)
                energy += elec_e

            if key in self.nuc_repulsion:
                nuc_repulsion = self.nuc_repulsion[key]
                nuc_e = nuc_repulsion(xyz=xyz,
                                      z=z,
                                      nbrs=nbrs,
                                      num_atoms=num_atoms)
                energy += nuc_e

            results.update({key: energy})

            if key in self.electrostatics:
                dipole = get_dipole(xyz=xyz,
                                    q=q,
                                    num_atoms=num_atoms)
                suffix = "_" + key.split("_")[-1]
                if not any([i.isdigit() for i in suffix]):
                    suffix = ""
                results.update({f"dipole{suffix}": dipole,
                                f"q{suffix}": q})

        return results

    def add_grad(self,
                 xyz,
                 grad_keys,
                 results):
        if grad_keys is None:
            grad_keys = self.grad_keys

        for key in grad_keys:
            base_key = key.replace("_grad", "")
            grad = compute_grad(inputs=xyz,
                                output=results[base_key])
            results[key] = grad

        return results

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
            f += y_t

        results = self.get_results(z=z,
                                   f=f,
                                   num_atoms=num_atoms,
                                   xyz=xyz,
                                   charge=charge,
                                   mol_nbrs=mol_nbrs,
                                   nbrs=nbrs)

        results = self.add_grad(xyz=xyz,
                                grad_keys=grad_keys,
                                results=results)

        return results

    def forward(self, *args, **kwargs):
        try:
            return self.fwd(*args, **kwargs)
        except Exception as e:
            print(e)
            import pdb
            pdb.post_mortem()
