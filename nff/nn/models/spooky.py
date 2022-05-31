import torch
from torch import nn
from nff.utils.scatter import compute_grad
from nff.utils.tools import make_directed
from nff.utils import constants as const
from nff.nn.modules.spooky import (DEFAULT_DROPOUT, DEFAULT_ACTIVATION,
                                   DEFAULT_MAX_Z, DEFAULT_RES_LAYERS,
                                   CombinedEmbedding, InteractionBlock,
                                   AtomwiseReadout, Electrostatics,
                                   NuclearRepulsion, get_dipole)
from nff.nn.modules.schnet import get_rij, get_offsets


from nff.nn.models.spooky_net_source.spookynet import SpookyNet as SourceSpooky


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
    """
    Simon's version of SpookyNet before the source code was released, which doesn't
    work properly
    """

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
                             residual_layers=residual_layers,
                             l_max=default(modelparams.get("l_max"), 2),
                             fast_feats=modelparams.get("fast_feats"))
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

        self.r_cut = r_cut

    def get_results(self,
                    z,
                    f,
                    num_atoms,
                    xyz,
                    charge,
                    nbrs,
                    offsets,
                    mol_offsets,
                    mol_nbrs):

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
                                           mol_nbrs=mol_nbrs,
                                           mol_offsets=mol_offsets)
                energy += elec_e

            if key in self.nuc_repulsion:
                nuc_repulsion = self.nuc_repulsion[key]
                nuc_e = nuc_repulsion(xyz=xyz,
                                      z=z,
                                      nbrs=nbrs,
                                      num_atoms=num_atoms,
                                      offsets=offsets)

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

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        interac = self.interactions[0]
        self.cutoff = interac.local.g_0.r_cut

    def fwd(self,
            batch,
            xyz=None,
            grad_keys=None):

        nxyz = batch['nxyz']
        nbrs, _ = make_directed(batch['nbr_list'])

        z = nxyz[:, 0].long()
        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        charge = batch['charge']
        spin = batch['spin']
        num_atoms = batch['num_atoms']
        offsets = get_offsets(batch, 'offsets')
        mol_offsets = get_offsets(batch, 'mol_offsets')
        mol_nbrs = batch.get('mol_nbrs')

        x = self.embedding(charge=charge,
                           spin=spin,
                           z=z,
                           num_atoms=num_atoms)

        # get r_ij including offsets and removing neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz,
                             batch=batch,
                             nbrs=nbrs,
                             cutoff=self.cutoff)
        f = torch.zeros_like(x)

        for i, interaction in enumerate(self.interactions):
            x, y_t = interaction(x=x,
                                 xyz=xyz,
                                 nbrs=nbrs,
                                 num_atoms=num_atoms,
                                 r_ij=r_ij)
            f = f + y_t

        results = self.get_results(z=z,
                                   f=f,
                                   num_atoms=num_atoms,
                                   xyz=xyz,
                                   charge=charge,
                                   nbrs=nbrs,
                                   offsets=offsets,
                                   mol_offsets=mol_offsets,
                                   mol_nbrs=mol_nbrs)

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


class RealSpookyNet(SourceSpooky):
    """
    Wrapper around the real source code for SpookyNet, so we can use it in NFF
    """

    def __init__(self,
                 params):

        super().__init__(**params)

        self.int_dtype = torch.long
        self.float_dtype = torch.float32

        self.to(self.float_dtype)
        self.output_key = params["output_key"]
        self.dip_key = params["dip_key"]
        self.charge_key = params["charge_key"]

    def get_full_nbrs(self,
                      batch):

        idx_i = batch['mol_nbrs'][:, 0]
        idx_j = batch['mol_nbrs'][:, 1]

        return idx_i, idx_j

    def get_regular_nbrs(self, batch):
        nbrs = batch['nbr_list']
        idx_i = nbrs[:, 0]
        idx_j = nbrs[:, 1]

        return idx_i, idx_j

    @property
    def device(self):
        return self.device

    @device.setter
    def device(self, val):
        self.to(val)

    def forward(self, batch):
        full_nbrs = any([self.use_d4_dispersion,
                         self.use_electrostatics])

        if full_nbrs:
            idx_i, idx_j = self.get_full_nbrs(batch)
            cell_offsets = batch.get('mol_offsets')
        else:
            idx_i, idx_j = self.get_regular_nbrs(batch)
            cell_offsets = batch.get('offsets')

        nxyz = batch['nxyz']
        xyz = nxyz[:, 1:].to(self.float_dtype)
        xyz.requires_grad = True
        device = xyz.device

        Z = nxyz[:, 0].to(self.int_dtype)

        num_atoms = batch['num_atoms']
        num_batch = len(num_atoms)
        batch_seg = torch.cat([torch.ones(int(num_atoms)) * i for i, num_atoms in
                               enumerate(num_atoms)]).to(self.int_dtype
                                                         ).to(device)

        out = super().forward(Z=Z,
                              Q=batch['charge'].to(self.float_dtype),
                              S=batch['spin'].to(self.float_dtype),
                              R=xyz,
                              idx_i=idx_i,
                              idx_j=idx_j,
                              num_batch=num_batch,
                              batch_seg=batch_seg,
                              cell=batch.get('cell'),
                              cell_offsets=cell_offsets)

        energy, forces, dipole, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = out

        grad_key = f"{self.output_key}_grad"

        results = {
            # energy in kcal/mol in NFF datasets
            self.output_key: energy * const.EV_TO_KCAL_MOL,
            # forces in kcal/mol/A in NFF datasets
            grad_key: -forces * const.EV_TO_KCAL_MOL,
            # dipole already given in eA in NFF datasets
            self.dip_key: dipole,
            "atom_features": f,
            "atomic_energies": ea * const.EV_TO_KCAL_MOL,
            self.charge_key: qa,
            "atomic_zbl": ea_rep * const.EV_TO_KCAL_MOL,
            "atom_vwd": ea_vdw * const.EV_TO_KCAL_MOL,
            "polarizabilities": pa,
            "c6": c6
        }

        return results
