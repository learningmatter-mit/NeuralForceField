import torch
from torch import nn
import copy
import numpy as np


from nff.nn.models.painn import Painn
from nff.nn.models.spooky import parse_add_ons
from nff.nn.modules.spooky_painn import MessageBlock as SpookyMessage
from nff.nn.modules.spooky_painn import Electrostatics as PainnElectrostatics
from nff.nn.modules.spooky_painn import CombinedEmbedding
from nff.nn.modules.painn import MessageBlock as PainnMessage
from nff.nn.modules.spooky import NuclearRepulsion
from nff.nn.modules.diabat import DiabaticReadout
from nff.nn.layers import Diagonalize
from nff.utils.tools import make_directed
from nff.utils.scatter import compute_grad

from nff.nn.modules.schnet import (AttentionPool, SumPool, MolFpPool,
                                   MeanPool, get_offsets, get_rij)

POOL_DIC = {"sum": SumPool,
            "mean": MeanPool,
            "attention": AttentionPool,
            "mol_fp": MolFpPool}


def default(dic, key, val):
    dic_val = dic.get(key)
    if dic_val is None:
        dic_val = val

    return dic_val


def get_elec_terms(modelparams):
    dic = dict(
        charge_charge=default(modelparams, "charge_charge", True),
        charge_dipole=default(modelparams, "charge_dipole", False),
        dipole_dipole=default(modelparams, "dipole_dipole", False),
        point_dipoles=default(modelparams, "point_dipoles", False)
    )

    return dic


class SpookyPainn(Painn):
    def __init__(self,
                 modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters

        """

        Painn.__init__(self,
                       modelparams)

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        learnable_k = modelparams.get("learnable_k", False)
        conv_dropout = modelparams.get("conv_dropout", 0)
        non_local = modelparams['non_local']

        add_ons = parse_add_ons(modelparams)
        add_nuc_keys, add_elec_keys, add_disp_keys = add_ons
        msg_class = SpookyMessage if non_local else PainnMessage

        self.embed_block = CombinedEmbedding(feat_dim=feat_dim,
                                             activation=activation)

        self.message_blocks = nn.ModuleList(
            [msg_class(feat_dim=feat_dim,
                       activation=activation,
                       n_rbf=n_rbf,
                       cutoff=cutoff,
                       learnable_k=learnable_k,
                       dropout=conv_dropout,
                       fast_feats=modelparams.get("fast_feats"))
             for _ in range(num_conv)]
        )

        elec_terms = get_elec_terms(modelparams)

        self.electrostatics = nn.ModuleDict({
            key: PainnElectrostatics(feat_dim=feat_dim,
                                     activation=activation,
                                     r_cut=cutoff,
                                     **elec_terms)
            for key in add_elec_keys
        })

        self.nuc_repulsion = nn.ModuleDict({
            key: NuclearRepulsion(r_cut=cutoff)
            for key in add_nuc_keys
        })

        if add_disp_keys:
            raise NotImplementedError("Dispersion not implemented")
        self.cutoff = cutoff

    def atomwise(self,
                 batch,
                 nbrs,
                 num_atoms,
                 xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip
                         for key in self.output_keys}

        nxyz = batch['nxyz']
        charge = batch['charge']
        spin = batch['spin']

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()
        # include offests

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz,
                             batch=batch,
                             nbrs=nbrs,
                             cutoff=self.cutoff)
        s_i, v_i = self.embed_block(charge=charge,
                                    spin=spin,
                                    z=z_numbers,
                                    num_atoms=num_atoms)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i,
                v_j=v_i,
                r_ij=r_ij,
                nbrs=nbrs,
                num_atoms=num_atoms.tolist())

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i,
                                                v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results['features'] = s_i

        return results, xyz, s_i, v_i

    def add_phys(self,
                 results,
                 s_i,
                 v_i,
                 xyz,
                 z,
                 charge,
                 nbrs,
                 num_atoms,
                 offsets,
                 mol_offsets,
                 mol_nbrs):

        electrostatics = getattr(self, "electrostatics", {})
        nuc_repulsion = getattr(self, "nuc_repulsion", {})

        for key in self.output_keys:
            if key in electrostatics:
                elec_module = self.electrostatics[key]
                elec_e, q, dip_atom, full_dip = elec_module(
                    s_i=s_i,
                    v_i=v_i,
                    z=z,
                    xyz=xyz,
                    total_charge=charge,
                    num_atoms=num_atoms,
                    mol_nbrs=mol_nbrs,
                    mol_offsets=mol_offsets)

                results[key] = results[key] + elec_e.reshape(-1)

            if key in nuc_repulsion:
                nuc_module = self.nuc_repulsion[key]
                nuc_e = nuc_module(xyz=xyz,
                                   z=z,
                                   nbrs=nbrs,
                                   num_atoms=num_atoms,
                                   offsets=offsets)
                results[key] = results[key] + nuc_e.reshape(-1)

            if key in electrostatics:
                suffix = "_" + key.split("_")[-1]
                if not any([i.isdigit() for i in suffix]):
                    suffix = ""
                results.update({f"dipole{suffix}": full_dip,
                                f"q{suffix}": q,
                                f"dip_atom{suffix}": dip_atom})

    def pool(self,
             batch,
             atomwise_out,
             xyz,
             nbrs,
             num_atoms,
             z,
             s_i,
             v_i):

        offsets = get_offsets(batch, 'offsets')
        mol_offsets = get_offsets(batch, 'mol_offsets')
        mol_nbrs = batch.get('mol_nbrs')

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0]
                                    .readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key
                             in self.output_keys}

        all_results = {}
        for key, pool_obj in self.pool_dic.items():
            results = pool_obj(batch=batch,
                               xyz=xyz,
                               atomwise_output=atomwise_out,
                               grad_keys=[],
                               out_keys=[key])
            all_results.update(results)

        self.add_phys(results=all_results,
                      s_i=s_i,
                      v_i=v_i,
                      xyz=xyz,
                      z=z,
                      charge=batch['charge'],
                      nbrs=nbrs,
                      num_atoms=num_atoms,
                      offsets=offsets,
                      mol_offsets=mol_offsets,
                      mol_nbrs=mol_nbrs)

        for key in self.grad_keys:
            output = all_results[key.replace("_grad", "")]
            grad = compute_grad(output=output,
                                inputs=xyz)
            all_results[key] = grad

        return all_results, xyz

    def run(self,
            batch,
            xyz=None,
            **kwargs):

        nbrs, _ = make_directed(batch['nbr_list'])
        num_atoms = batch['num_atoms']
        z = batch['nxyz'][:, 0].long()

        atomwise_out, xyz, s_i, v_i = self.atomwise(
            batch=batch,
            xyz=xyz,
            nbrs=nbrs,
            num_atoms=num_atoms)
        all_results, xyz = self.pool(batch=batch,
                                     atomwise_out=atomwise_out,
                                     xyz=xyz,
                                     nbrs=nbrs,
                                     num_atoms=num_atoms,
                                     z=z,
                                     s_i=s_i,
                                     v_i=v_i)

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        return all_results, xyz


def get_others_to_eig(diabat_keys):
    others_to_eig = copy.deepcopy(diabat_keys)
    num_states = len(diabat_keys)
    for i in range(num_states):
        for j in range(num_states):
            val = others_to_eig[i][j]
            others_to_eig[i][j] = "dipole_" + val.split("_")[-1]
    return others_to_eig


class SpookyPainnDiabat(SpookyPainn):

    def __init__(self, modelparams):
        """
        `diabat_keys` has the shape of a 2x2 matrix
        """

        energy_keys = modelparams["output_keys"]
        diabat_keys = modelparams["diabat_keys"]
        new_out_keys = list(set(np.array(diabat_keys).reshape(-1)
                                .tolist()))

        new_modelparams = copy.deepcopy(modelparams)
        new_modelparams.update({"output_keys": new_out_keys,
                                "grad_keys": []})
        super().__init__(new_modelparams)

        self.diag = Diagonalize()
        others_to_eig = ([get_others_to_eig(diabat_keys)]
                         if self.electrostatics else None)

        self.diabatic_readout = DiabaticReadout(
            diabat_keys=diabat_keys,
            grad_keys=modelparams["grad_keys"],
            energy_keys=energy_keys,
            delta=False,
            stochastic_dic=modelparams.get("stochastic_dic"),
            cross_talk_dic=modelparams.get("cross_talk_dic"),
            hellmann_feynman=modelparams.get("hellmann_feynman", True),
            others_to_eig=others_to_eig)
        self.add_nacv = modelparams.get("add_nacv", False)
        self.diabat_keys = diabat_keys
        self.off_diag_keys = self.get_off_diag_keys()

    @property
    def _grad_keys(self):
        return self.grad_keys

    @_grad_keys.setter
    def _grad_keys(self, value):
        self.grad_keys = value
        self.diabatic_readout.grad_keys = value

    def get_off_diag_keys(self):
        num_states = len(self.diabat_keys)
        off_diag = []
        for i in range(num_states):
            for j in range(num_states):
                if j <= i:
                    continue
                off_diag.append(self.diabat_keys[i][j])
        return off_diag

    def get_diabat_charge(self,
                          key,
                          charge):

        if key in self.off_diag_keys:
            total_charge = torch.zeros_like(charge)
        else:
            total_charge = charge
        return total_charge

    def add_phys(self,
                 results,
                 s_i,
                 v_i,
                 xyz,
                 z,
                 charge,
                 nbrs,
                 num_atoms,
                 offsets,
                 mol_offsets,
                 mol_nbrs):
        """
        Over-write because transition charges must sum to 0, not
        to the total charge
        """

        electrostatics = getattr(self, "electrostatics", {})
        nuc_repulsion = getattr(self, "nuc_repulsion", {})

        for key in self.output_keys:
            if key in electrostatics:
                elec_module = self.electrostatics[key]

                # transition charges sum to 0

                total_charge = self.get_diabat_charge(key=key,
                                                      charge=charge)

                mol_nbrs, _ = make_undirected(batch['mol_nbrs'])
                elec_e, q, dip_atom, full_dip = elec_module(
                    s_i=s_i,
                    v_i=v_i,
                    z=z,
                    xyz=xyz,
                    total_charge=charge,
                    num_atoms=num_atoms,
                    mol_nbrs=mol_nbrs,
                    mol_offsets=mol_offsets)

                results[key] = results[key] + elec_e.reshape(-1)

            if key in nuc_repulsion:
                nuc_module = self.nuc_repulsion[key]
                nuc_e = nuc_module(xyz=xyz,
                                   z=z,
                                   nbrs=nbrs,
                                   num_atoms=num_atoms,
                                   offsets=offsets)
                results[key] = results[key] + nuc_e.reshape(-1)

            if key in electrostatics:
                suffix = "_" + key.split("_")[-1]
                if not any([i.isdigit() for i in suffix]):
                    suffix = ""
                results.update({f"dipole{suffix}": full_dip,
                                f"q{suffix}": q,
                                f"dip_atom{suffix}": dip_atom})

    def forward(self,
                batch,
                xyz=None,
                add_nacv=False,
                add_grad=True,
                add_gap=True):

        # for backwards compatability
        self.grad_keys = []

        if not hasattr(self, "output_keys"):
            diabat_keys = self.diabatic_readout.diabat_keys
            self.output_keys = list(set(np.array(diabat_keys)
                                        .reshape(-1)
                                        .tolist()))
        if hasattr(self, "add_nacv"):
            add_nacv = self.add_nacv

        diabat_results, xyz = self.run(batch=batch,
                                       xyz=xyz)
        results = self.diabatic_readout(batch=batch,
                                        xyz=xyz,
                                        results=diabat_results,
                                        add_nacv=add_nacv,
                                        add_grad=add_grad,
                                        add_gap=add_gap)

        return results
