import copy
import os
import sys
from collections import Counter

import nff.utils.constants as const
import numpy as np
import torch
from functorch import combine_state_for_ensemble, vmap
from nff.data import Dataset, collate_dicts
from nff.data.sparse import sparsify_array
from nff.nn.graphop import split_and_sum
from nff.nn.models.cp3d import OnlyBondUpdateCP3D
from nff.nn.models.hybridgraph import HybridGraphConv
from nff.nn.models.schnet import SchNet, SchNetDiabat
from nff.nn.models.schnet_features import SchNetFeatures
from nff.nn.utils import (clean_matrix, lattice_points_in_supercell,
                          torch_nbr_list)
from nff.train.builders.model import load_model
from nff.utils.constants import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL
from nff.utils.cuda import batch_to
from nff.utils.geom import batch_compute_distance, compute_distances
from nff.utils.scatter import compute_grad
from torch.autograd import grad

from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress

HARTREE_TO_EV = HARTREE_TO_KCAL_MOL / EV_TO_KCAL_MOL
# from torch.func import functional_call, stack_module_state


DEFAULT_CUTOFF = 5.0
DEFAULT_DIRECTED = False
DEFAULT_SKIN = 1.0
UNDIRECTED = [SchNet, SchNetDiabat, HybridGraphConv, SchNetFeatures, OnlyBondUpdateCP3D]


def check_directed(model, atoms):
    model_cls = model.__class__.__name__
    msg = f"{model_cls} needs a directed neighbor list"
    assert atoms.directed, msg


class AtomsBatch(Atoms):
    """Class to deal with the Neural Force Field and batch several
    Atoms objects.
    """

    def __init__(
        self,
        *args,
        props=None,
        cutoff=DEFAULT_CUTOFF,
        directed=DEFAULT_DIRECTED,
        requires_large_offsets=False,
        cutoff_skin=DEFAULT_SKIN,
        dense_nbrs=True,
        device=0,
        **kwargs,
    ):
        """

        Args:
            *args: Description
            nbr_list (None, optional): Description
            pbc_index (None, optional): Description
            cutoff (TYPE, optional): Description
            cutoff_skin (float): extra distance added to cutoff
                            to ensure we don't miss neighbors between nbr
                            list updates.
            **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        if props is None:
            props = {}

        self.props = props
        self.nbr_list = props.get("nbr_list", None)
        self.offsets = props.get("offsets", None)
        self.directed = directed
        self.num_atoms = props.get("num_atoms", torch.LongTensor([len(self)])).reshape(
            -1
        )
        self.props["num_atoms"] = self.num_atoms
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.requires_large_offsets = requires_large_offsets
        if dense_nbrs:
            self.mol_nbrs, self.mol_idx = self.get_mol_nbrs()
        else:
            self.mol_nbrs, self.mol_idx = None, None

    def convert_props_units(self, target_unit):
        """Converts the units of the properties to the desired unit.
        Args:
            target_unit (str): target unit.
        """
        conversion_factors = {
            ("eV", "kcal/mol"): const.EV_TO_KCAL,
            ("eV", "atomic"): const.EV_TO_AU,
            ("kcal/mol", "eV"): const.KCAL_TO_EV,
            ("kcal/mol", "atomic"): const.KCAL_TO_AU,
            ("atomic", "eV"): const.AU_TO_EV,
            ("atomic", "kcal/mol"): const.AU_TO_KCAL,
        }

        if target_unit not in ["kcal/mol", "eV", "atomic"]:
            raise NotImplementedError(f"Unit {target_unit} not implemented")

        curr_unit = self.props.get("units", "eV")

        if target_unit == curr_unit:
            return

        conversion_factor = conversion_factors.get((curr_unit, target_unit))
        if conversion_factor is None:
            raise NotImplementedError(
                f"Conversion from {curr_unit} to {target_unit} not implemented"
            )

        self.props = const.convert_units(self.props, conversion_factor)
        self.props.update({"units": target_unit})
        return

    def get_mol_nbrs(self, r_cut=95):
        """
        Dense directed neighbor list for each molecule, in case that's needed
        in the model calculation
        """

        # periodic systems
        if np.array([atoms.pbc.any() for atoms in self.get_list_atoms()]).any():
            nbrs = []
            nbrs_T = []
            nbrs = []
            z = []
            N = []
            lattice_points = []
            mask_applied = []
            _xyzs = []
            xyz_T = []
            num_atoms = []
            for atoms in self.get_list_atoms():
                nxyz = np.concatenate(
                    [
                        atoms.get_atomic_numbers().reshape(-1, 1),
                        atoms.get_positions().reshape(-1, 3),
                    ],
                    axis=1,
                )
                _xyz = torch.from_numpy(nxyz[:, 1:])
                # only works if the cell for all crystals in batch are the same
                cell = atoms.get_cell()

                # cutoff specified by r_cut in Bohr (a.u.)
                # estimate getting close to the cutoff with supercell expansion
                a_mul = int(
                    np.ceil((r_cut * const.BOHR_RADIUS) / np.linalg.norm(cell[0]))
                )
                b_mul = int(
                    np.ceil((r_cut * const.BOHR_RADIUS) / np.linalg.norm(cell[1]))
                )
                c_mul = int(
                    np.ceil((r_cut * const.BOHR_RADIUS) / np.linalg.norm(cell[2]))
                )
                supercell_matrix = np.array(
                    [[a_mul, 0, 0], [0, b_mul, 0], [0, 0, c_mul]]
                )
                supercell = clean_matrix(supercell_matrix @ cell)

                # cartesian lattice points
                lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
                _lattice_points = np.dot(lattice_points_frac, supercell)

                # need to get all negative lattice translation vectors
                # but remove duplicate 0 vector
                zero_idx = np.where(
                    np.all(_lattice_points.__eq__(np.array([0, 0, 0])), axis=1)
                )[0][0]
                _lattice_points = np.concatenate(
                    [_lattice_points[zero_idx:, :], _lattice_points[:zero_idx, :]]
                )

                _z = torch.from_numpy(nxyz[:, 0]).long().to(self.device)
                _N = len(_lattice_points)
                # perform lattice translations on positions
                lattice_points_T = (
                    torch.tile(
                        torch.from_numpy(_lattice_points),
                        ((len(_xyz),) + (1,) * (len(_lattice_points.shape) - 1)),
                    )
                    / const.BOHR_RADIUS
                ).to(self.device)
                _xyz_T = (
                    torch.repeat_interleave(_xyz, _N, dim=0) / const.BOHR_RADIUS
                ).to(self.device)
                _xyz_T = _xyz_T + lattice_points_T

                # get valid indices within the cutoff
                num = _xyz.shape[0]
                idx = torch.arange(num)
                x, y = torch.meshgrid(idx, idx)
                _nbrs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1).to(
                    self.device
                )
                _lattice_points = torch.tile(
                    torch.from_numpy(_lattice_points).to(self.device),
                    ((len(_nbrs),) + (1,) * (len(_lattice_points.shape) - 1)),
                )

                # convert everything from Angstroms to Bohr
                _xyz = _xyz / const.BOHR_RADIUS
                _lattice_points = _lattice_points / const.BOHR_RADIUS

                _nbrs_T = torch.repeat_interleave(_nbrs, _N, dim=0).to(self.device)
                # ensure that A != B when T=0
                # since first index in _lattice_points corresponds to T=0
                # get the idxs on which to apply the mask
                idxs_to_apply = torch.tensor([True] * len(_nbrs_T)).to(self.device)
                idxs_to_apply[::_N] = False
                # get the mask that we want to apply
                mask = _nbrs_T[:, 0] != _nbrs_T[:, 1]
                # do a joint boolean operation to get the mask
                _mask_applied = torch.logical_or(idxs_to_apply, mask)
                _nbrs_T = _nbrs_T[_mask_applied]
                _lattice_points = _lattice_points[_mask_applied]

                nbrs_T.append(_nbrs_T)
                nbrs.append(_nbrs)
                z.append(_z)
                N.append(_N)
                lattice_points.append(_lattice_points)
                mask_applied.append(_mask_applied)
                xyz_T.append(_xyz_T)
                _xyzs.append(_xyz)

                num_atoms.append(len(_xyz))

            nbrs_info = (nbrs_T, nbrs, z, N, lattice_points, mask_applied)

            mol_idx = torch.cat(
                [torch.zeros(num) + i for i, num in enumerate(num_atoms)]
            ).long()

            return nbrs_info, mol_idx

        # non-periodic systems
        else:
            counter = 0
            nbrs = []

            for atoms in self.get_list_atoms():
                nxyz = np.concatenate(
                    [
                        atoms.get_atomic_numbers().reshape(-1, 1),
                        atoms.get_positions().reshape(-1, 3),
                    ],
                    axis=1,
                )

                n = nxyz.shape[0]
                idx = torch.arange(n)
                x, y = torch.meshgrid(idx, idx)

                # undirected neighbor list
                these_nbrs = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)
                these_nbrs = these_nbrs[these_nbrs[:, 0] != these_nbrs[:, 1]]

                nbrs.append(these_nbrs + counter)
                counter += n

            nbrs = torch.cat(nbrs)
            mol_idx = torch.cat(
                [torch.zeros(num) + i for i, num in enumerate(self.num_atoms)]
            ).long()

            return nbrs, mol_idx

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
           inside the unit cell of the system.
        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                             of the atoms.
        """
        nxyz = np.concatenate(
            [
                self.get_atomic_numbers().reshape(-1, 1),
                self.get_positions().reshape(-1, 3),
            ],
            axis=1,
        )

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
        to be sent to the model.
        Returns:
           batch (dict): batch with the keys 'nxyz',
                         'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()

        self.props["nbr_list"] = self.nbr_list
        self.props["offsets"] = self.offsets
        if self.pbc.any():
            self.props["cell"] = torch.Tensor(np.array(self.cell))
            self.props["lattice"] = self.cell.tolist()

        self.props["nxyz"] = torch.Tensor(self.get_nxyz())
        if self.props.get("num_atoms") is None:
            self.props["num_atoms"] = torch.LongTensor([len(self)])

        if self.mol_nbrs is not None:
            self.props["mol_nbrs"] = self.mol_nbrs

        if self.mol_idx is not None:
            self.props["mol_idx"] = self.mol_idx

        return self.props

    def get_list_atoms(self):
        if self.props.get("num_atoms") is None:
            self.props["num_atoms"] = torch.LongTensor([len(self)])

        mol_split_idx = self.props["num_atoms"].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))
        masses = list(torch.Tensor(self.get_masses()).split(mol_split_idx))

        # split cell if periodic
        if self.pbc.any():
            if "lattice" in self.props:
                cells = torch.split(torch.Tensor(self.props['lattice']), 3)
            else:
                cells = torch.unsqueeze(torch.Tensor(np.array(self.cell)), 0).repeat(len(mol_split_idx), 1, 1)
        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            atoms = Atoms(Z[i].tolist(),
                          molecule_xyz.numpy(),
                          cell=cells[i].numpy() if self.pbc.any() else None,
                          pbc=self.pbc)

            # in case you artificially changed the masses
            # of any of the atoms
            atoms.set_masses(masses[i])

            Atoms_list.append(atoms)

        return Atoms_list

    def update_num_atoms(self):
        self.props["num_atoms"] = torch.tensor([len(self)])

    def update_nbr_list(self, update_atoms=False):
        """Update neighbor list and the periodic reindexing
        for the given Atoms object.
        Args:
        cutoff(float): maximum cutoff for which atoms are
                                       considered interacting.
        Returns:
        nbr_list(torch.LongTensor)
        offsets(torch.Tensor)
        nxyz(torch.Tensor)
        """
        if update_atoms:
            self.update_num_atoms()

        Atoms_list = self.get_list_atoms()

        ensemble_nbr_list = []
        ensemble_offsets_list = []

        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to, offsets = torch_nbr_list(
                atoms,
                (self.cutoff + self.cutoff_skin),
                device=self.device,
                directed=self.directed,
                requires_large_offsets=self.requires_large_offsets,
            )

            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            these_offsets = sparsify_array(offsets.dot(self.get_cell()))

            # non-periodic
            if isinstance(these_offsets, int):
                these_offsets = torch.Tensor(offsets)

            ensemble_nbr_list.append(self.props["num_atoms"][:i].sum() + nbr_list)
            ensemble_offsets_list.append(these_offsets)

        ensemble_nbr_list = torch.cat(ensemble_nbr_list)

        if all([isinstance(i, int) for i in ensemble_offsets_list]):
            ensemble_offsets_list = torch.Tensor(ensemble_offsets_list)
        else:
            ensemble_offsets_list = torch.cat(ensemble_offsets_list)

        self.nbr_list = ensemble_nbr_list
        self.offsets = ensemble_offsets_list

        return ensemble_nbr_list, ensemble_offsets_list

    def get_embedding(self):
        embedding = self._calc.get_embedding(self)
        mol_split_idx = self.props["num_atoms"].tolist()
        batch_embedding = torch.stack(
            torch.Tensor(embedding).split(mol_split_idx)
        ).numpy()

        return batch_embedding

    def get_batch_energies(self):
        if self._calc is None:
            raise RuntimeError("Atoms object has no calculator.")

        if not hasattr(self._calc, "get_potential_energies"):
            raise RuntimeError(
                "The calculator for atomwise energies is not implemented"
            )

        energies = self.get_potential_energies()

        batched_energies = split_and_sum(
            torch.Tensor(energies), self.props["num_atoms"].tolist()
        )

        return batched_energies.detach().cpu().numpy()

    def get_batch_kinetic_energy(self):
        if self.get_momenta().any():
            atomwise_ke = torch.Tensor(
                0.5 * self.get_momenta() * self.get_velocities()
            ).sum(-1)
            batch_ke = split_and_sum(atomwise_ke, self.props["num_atoms"].tolist())
            return batch_ke.detach().cpu().numpy()

        else:
            print("No momenta are set for atoms")

    def get_batch_T(self):
        T = self.get_batch_kinetic_energy() / (
            1.5 * units.kB * self.props["num_atoms"].detach().cpu().numpy()
        )
        return T

    def batch_properties():
        pass

    def batch_virial():
        pass

    @classmethod
    def from_atoms(cls, atoms, **kwargs):
        return cls(
            atoms, positions=atoms.positions, numbers=atoms.numbers, props={}, **kwargs
        )


class BulkPhaseMaterials(Atoms):
    """Class to deal with the Neural Force Field and batch molecules together
    in a box for handling boxphase.
    """

    def __init__(
        self,
        *args,
        props={},
        cutoff=DEFAULT_CUTOFF,
        nbr_torch=False,
        device="cpu",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        """

        Args:
        *args: Description
        nbr_list (None, optional): Description
        pbc_index (None, optional): Description
        cutoff (TYPE, optional): Description
        **kwargs: Description
        """
        super().__init__(*args, **kwargs)

        self.props = props
        self.nbr_list = self.props.get("nbr_list", None)
        self.offsets = self.props.get("offsets", None)
        self.num_atoms = self.props.get("num_atoms", None)
        self.cutoff = cutoff
        self.nbr_torch = nbr_torch
        self.device = device
        self.directed = directed

    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
           inside the unit cell of the system.
        Returns:
                nxyz (np.array): atomic numbers + cartesian coordinates
                                                 of the atoms.
        """
        nxyz = np.concatenate(
            [
                self.get_atomic_numbers().reshape(-1, 1),
                self.get_positions().reshape(-1, 3),
            ],
            axis=1,
        )

        return nxyz

    def get_batch(self):
        """Uses the properties of Atoms to create a batch
           to be sent to the model.

        Returns:
           batch (dict): batch with the keys 'nxyz',
           'num_atoms', 'nbr_list' and 'offsets'
        """

        if self.nbr_list is None or self.offsets is None:
            self.update_nbr_list()
            self.props["nbr_list"] = self.nbr_list
            self.props["atoms_nbr_list"] = self.atoms_nbr_list
            self.props["offsets"] = self.offsets

        self.props["nbr_list"] = self.nbr_list
        self.props["atoms_nbr_list"] = self.atoms_nbr_list
        self.props["offsets"] = self.offsets
        self.props["nxyz"] = torch.Tensor(self.get_nxyz())

        return self.props

    def update_system_nbr_list(self, cutoff, exclude_atoms_nbr_list=True):
        """Update undirected neighbor list and the periodic reindexing
        for the given Atoms object.

        Args:
        cutoff (float): maximum cutoff for which atoms are
        considered interacting.

        Returns:
        nbr_list (torch.LongTensor)
        offsets (torch.Tensor)
             nxyz (torch.Tensor)
        """

        if self.nbr_torch:
            edge_from, edge_to, offsets = torch_nbr_list(
                self, self.cutoff, device=self.device
            )
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
        else:
            edge_from, edge_to, offsets = neighbor_list("ijS", self, self.cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
            if not getattr(self, "directed", DEFAULT_DIRECTED):
                offsets = offsets[nbr_list[:, 1] > nbr_list[:, 0]]
                nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

        if exclude_atoms_nbr_list:
            offsets_mat = torch.zeros(len(self), len(self), 3)
            nbr_list_mat = torch.zeros(len(self), len(self)).to(torch.long)
            atom_nbr_list_mat = torch.zeros(len(self), len(self)).to(torch.long)

            offsets_mat[nbr_list[:, 0], nbr_list[:, 1]] = offsets
            nbr_list_mat[nbr_list[:, 0], nbr_list[:, 1]] = 1
            atom_nbr_list_mat[self.atoms_nbr_list[:, 0], self.atoms_nbr_list[:, 1]] = 1

            nbr_list_mat = nbr_list_mat - atom_nbr_list_mat
            nbr_list = nbr_list_mat.nonzero()
            offsets = offsets_mat[nbr_list[:, 0], nbr_list[:, 1], :]

        self.nbr_list = nbr_list
        self.offsets = sparsify_array(offsets.matmul(torch.Tensor(self.get_cell())))

    def get_list_atoms(self):
        mol_split_idx = self.props["num_subgraphs"].tolist()

        positions = torch.Tensor(self.get_positions())
        Z = torch.LongTensor(self.get_atomic_numbers())

        positions = list(positions.split(mol_split_idx))
        Z = list(Z.split(mol_split_idx))

        Atoms_list = []

        for i, molecule_xyz in enumerate(positions):
            Atoms_list.append(
                Atoms(Z[i].tolist(), molecule_xyz.numpy(), cell=self.cell, pbc=self.pbc)
            )

        return Atoms_list

    def update_atoms_nbr_list(self, cutoff):
        Atoms_list = self.get_list_atoms()

        intra_nbr_list = []
        for i, atoms in enumerate(Atoms_list):
            edge_from, edge_to = neighbor_list("ij", atoms, cutoff)
            nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))

            if not self.directed:
                nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

            intra_nbr_list.append(self.props["num_subgraphs"][:i].sum() + nbr_list)

        intra_nbr_list = torch.cat(intra_nbr_list)
        self.atoms_nbr_list = intra_nbr_list

    def update_nbr_list(self):
        self.update_atoms_nbr_list(self.props["atoms_cutoff"])
        self.update_system_nbr_list(self.props["system_cutoff"])


class NeuralFF(Calculator):
    """ASE calculator using a pretrained NeuralFF model"""

    implemented_properties = ["energy", "forces", "stress", "embedding"]

    def __init__(
        self,
        model,
        device="cpu",
        jobdir=None,
        en_key="energy",
        properties=["energy", "forces"],
        model_kwargs=None,
        **kwargs,
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model (TYPE): Description
        device (str): device on which the calculations will be performed
        properties (list of str): 'energy', 'forces' or both and also stress for only
        schnet  and painn
        **kwargs: Description
        model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)
        self.jobdir = jobdir
        self.en_key = en_key
        self.properties = properties
        self.model_kwargs = model_kwargs

    def to(self, device):
        self.device = device
        self.model.to(device)

    def log_embedding(self, jobdir, log_filename, props):
        """For the purposes of logging the NN embedding on-the-fly, to help with
        sampling after calling NFF on geometries."""

        log_file = os.path.join(jobdir, log_filename)
        if os.path.exists(log_file):
            log = np.load(log_file)
        else:
            log = None

        if log is not None:
            log = np.append(log, props[None, :, :, :], axis=0)
        else:
            log = props[None, :, :, :]

        np.save(log_filename, log)

        return

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        system_changes (default from ase)
        """

        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        requires_embedding = "embedding" in self.properties
        if requires_embedding:
            kwargs["requires_embedding"] = True
        if requires_stress:
            kwargs["requires_stress"] = True
        kwargs["grimme_disp"] = False
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        prediction = self.model(batch, **kwargs)

        # change energy and force to numpy array

        energy = prediction[self.en_key].detach().cpu().numpy() * (
            1 / const.EV_TO_KCAL_MOL
        )

        if grad_key in prediction:
            energy_grad = prediction[grad_key].detach().cpu().numpy() * (
                1 / const.EV_TO_KCAL_MOL
            )

        self.results = {"energy": energy.reshape(-1)}
        if "e_disp" in prediction:
            self.results["energy"] = self.results["energy"] + prediction["e_disp"]

        if "forces" in self.properties:
            self.results["forces"] = -energy_grad.reshape(-1, 3)
            if "forces_disp" in prediction:
                self.results["forces"] = (
                    self.results["forces"] + prediction["forces_disp"]
                )

        if requires_embedding:
            embedding = prediction["embedding"].detach().cpu().numpy()
            self.results["embedding"] = embedding

        if requires_stress:
            stress = prediction["stress_volume"].detach().cpu().numpy() * (
                1 / const.EV_TO_KCAL_MOL
            )
            self.results["stress"] = stress * (1 / atoms.get_volume())
            if "stress_disp" in prediction:
                self.results["stress"] = (
                    self.results["stress"] + prediction["stress_disp"]
                )
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])

    def get_embedding(self, atoms=None):
        return self.get_property("embedding", atoms)

    @classmethod
    def from_file(cls, model_path, device="cuda", **kwargs):
        model = load_model(model_path, **kwargs)
        out = cls(model=model, device=device, **kwargs)
        return out


class EnsembleNFF(Calculator):
    """Produces an ensemble of NFF calculators to predict the
    discrepancy between the properties"""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        models: list,
        device="cpu",
        jobdir=None,
        properties=["energy", "forces"],
        model_kwargs=None,
        **kwargs,
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
        model(TYPE): Description
        device(str): device on which the calculations will be performed
        **kwargs: Description
        model(one of nff.nn.models)

        """

        Calculator.__init__(self, **kwargs)
        self.models = models
        for m in self.models:
            m.eval()
        self.device = device
        self.to(device)
        self.jobdir = jobdir
        self.offset_data = {}
        self.properties = properties
        self.model_kwargs = model_kwargs

    def to(self, device):
        self.device = device
        for m in self.models:
            m.to(device)

    def offset_energy(self, atoms, energy: float or np.ndarray):
        """Offset the NFF energy to obtain the correct reference energy

        Args:
        atoms (Atoms): Atoms object
        energy (float or np.ndarray): energy to be offset
        """

        ads_count = Counter(atoms.get_chemical_symbols())

        stoidict = self.offset_data.get("stoidict", {})
        ref_en = 0
        for ele, num in ads_count.items():
            ref_en += num * stoidict.get(ele, 0.0)
        ref_en += stoidict.get("offset", 0.0)

        energy += ref_en * HARTREE_TO_EV
        return energy

    def log_ensemble(self, jobdir, log_filename, props):
        """For the purposes of logging the std on-the-fly, to help with
        sampling after calling NFF on geometries with high uncertainty."""

        props = np.swapaxes(np.expand_dims(props, axis=-1), 0, -1)

        log_file = os.path.join(jobdir, log_filename)
        if os.path.exists(log_file):
            log = np.load(log_file)
        else:
            log = None

        if log is not None:
            log = np.concatenate([log, props], axis=0)
        else:
            log = props

        np.save(log_filename, log)

        return

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
        atoms (AtomsBatch): custom Atoms subclass that contains implementation
            of neighbor lists, batching and so on. Avoids the use of the Dataset
            to calculate using the models created.
        properties (list of str): 'energy', 'forces' or both
        system_changes (default from ase)
        """

        for model in self.models:
            if not any([isinstance(model, i) for i in UNDIRECTED]):
                check_directed(model, atoms)

        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, properties, system_changes)

        # TODO can probably make it more efficient by only calculating the system changes

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        # TODO update atoms only when necessary
        atoms.update_nbr_list(update_atoms=True)

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        kwargs["grimme_disp"] = False
        if getattr(self, "model_kwargs", None) is not None:
            kwargs.update(self.model_kwargs)

        # run model
        # atomsbatch = AtomsBatch(atoms)
        # batch_to(atomsbatch.get_batch(), self.device)
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        batch["energy"] = []
        if "forces" in properties:
            batch["energy_grad"] = []

        energies = []
        gradients = []
        stresses = []

        # do this in parallel, using the version here: https://pytorch.org/functorch/1.13/notebooks/ensembling.html

        # fmodel, params, buffers = combine_state_for_ensemble(self.models)
        # predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, batch)

        # Construct a "stateless" version of one of the models. It is "stateless" in
        # the sense that the parameters are meta Tensors and do not have storage.
        # self.base_model = copy.deepcopy(self.models[0])
        # self.base_model = self.base_model.to('meta')
        # def fmodel(params, buffers, x):
        #     return functional_call(self.base_model, (params, buffers), (x,))

        # predictions_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, batch)

        # do this in parallel

        for model in self.models:
            prediction = model(batch, **kwargs)

            # change energy and force to numpy array
            energies.append(
                prediction["energy"].detach().cpu().numpy() * (1 / const.EV_TO_KCAL_MOL)
            )
            gradients.append(
                prediction["energy_grad"].detach().cpu().numpy()
                * (1 / const.EV_TO_KCAL_MOL)
            )
            if "stress_volume" in prediction:
                stresses.append(
                    prediction["stress_volume"].detach().cpu().numpy()
                    * (1 / const.EV_TO_KCAL_MOL)
                    * (1 / atoms.get_volume())
                )

        energies = np.stack(energies)
        gradients = np.stack(gradients)

        # offset energies
        energies = self.offset_energy(atoms, energies)

        if len(stresses) > 0:
            stresses = np.stack(stresses)

        self.results = {
            "energy": energies.mean(0).reshape(-1),
            "energy_std": energies.std(0).reshape(-1),
        }
        if "e_disp" in prediction:
            self.results["energy"] = self.results["energy"] + prediction["e_disp"]
        if self.jobdir is not None and system_changes:
            energy_std = self.results["energy_std"][None]
            self.log_ensemble(self.jobdir, "energy_nff_ensemble.npy", energies)

        if "forces" in properties:
            self.results["forces"] = -gradients.mean(0).reshape(-1, 3)
            self.results["forces_std"] = gradients.std(0).reshape(-1, 3)
            if "forces_disp" in prediction:
                self.results["forces"] = (
                    self.results["forces"] + prediction["forces_disp"]
                )
            if self.jobdir is not None:
                forces_std = self.results["forces_std"][None, :, :]
                self.log_ensemble(self.jobdir, "forces_nff_ensemble.npy", -gradients)

        if "stress" in properties:
            self.results["stress"] = stresses.mean(0)
            self.results["stress_std"] = stresses.std(0)
            if "stress_disp" in prediction:
                self.results["stress"] = (
                    self.results["stress"] + prediction["stress_disp"]
                )
            if self.jobdir is not None:
                stress_std = self.results["stress_std"][None, :, :]
                self.log_ensemble(self.jobdir, "stress_nff_ensemble.npy", stresses)

        atoms.results = self.results.copy()

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""
        Calculator.set(self, **kwargs)
        if "offset_data" in self.parameters.keys():
            self.offset_data = self.parameters["offset_data"]
            print(f"offset data: {self.offset_data} is set from parameters")

    @classmethod
    def from_files(cls, model_paths: list, device="cuda", **kwargs):
        models = [load_model(path) for path in model_paths]
        return cls(models, device, **kwargs)


class NeuralOptimizer:
    def __init__(self, optimizer, nbrlist_update_freq=5):
        self.optimizer = optimizer
        self.update_freq = nbrlist_update_freq

    def run(self, fmax=0.2, steps=1000):
        epochs = steps // self.update_freq

        for step in range(epochs):
            self.optimizer.run(fmax=fmax, steps=self.update_freq)
            self.optimizer.atoms.update_nbr_list()


class NeuralMetadynamics(NeuralFF):
    def __init__(
        self,
        model,
        pushing_params,
        old_atoms=None,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        NeuralFF.__init__(
            self,
            model=model,
            device=device,
            en_key=en_key,
            directed=DEFAULT_DIRECTED,
            **kwargs,
        )

        self.pushing_params = pushing_params
        self.old_atoms = old_atoms if (old_atoms is not None) else []
        self.steps_from_old = []

        # only apply the bias to certain atoms
        self.exclude_atoms = torch.LongTensor(
            self.pushing_params.get("exclude_atoms", [])
        )
        self.keep_idx = None

    def get_keep_idx(self, atoms):
        # correct for atoms not in the biasing potential

        if self.keep_idx is not None:
            assert len(self.keep_idx) + len(self.exclude_atoms) == len(atoms)
            return self.keep_idx

        keep_idx = torch.LongTensor(
            [i for i in range(len(atoms)) if i not in self.exclude_atoms]
        )
        self.keep_idx = keep_idx
        return keep_idx

    def make_dsets(self, atoms):
        keep_idx = self.get_keep_idx(atoms)
        # put the current atoms as the second dataset because that's the one
        # that gets its positions rotated and its gradient computed
        props_1 = {"nxyz": [torch.Tensor(atoms.get_nxyz())[keep_idx, :]]}
        props_0 = {
            "nxyz": [
                torch.Tensor(old_atoms.get_nxyz())[keep_idx, :]
                for old_atoms in self.old_atoms
            ]
        }

        dset_0 = Dataset(props_0, do_copy=False)
        dset_1 = Dataset(props_1, do_copy=False)

        return dset_0, dset_1

    def rmsd_prelims(self, atoms):
        num_atoms = len(atoms)

        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = 2 / (1 + torch.exp(-kappa * steps_from_old)) - 1

        # given in mHartree / atom in CREST paper
        k_i = self.pushing_params["k_i"] / 1000 * units.Hartree * num_atoms

        # given in Bohr^(-2) in CREST paper
        alpha_i = self.pushing_params["alpha_i"] / units.Bohr**2

        dsets = self.make_dsets(atoms)

        return k_i, alpha_i, dsets, f_damp

    def rmsd_push(self, atoms):
        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), 0.0

        k_i, alpha_i, dsets, f_damp = self.rmsd_prelims(atoms)

        delta_i, _, xyz_list = compute_distances(
            dataset=dsets[0],
            # do this on CPU - it's a small RMSD
            # and gradient calculation, so the
            # dominant time is data transfer to GPU.
            # Testing it out confirms that you get a
            # big slowdown from doing it on GPU
            device="cpu",
            # device=self.device,
            dataset_1=dsets[1],
            store_grad=True,
            collate_dicts=collate_dicts,
        )

        v_bias = (f_damp * k_i * torch.exp(-alpha_i * delta_i.reshape(-1) ** 2)).sum()

        f_bias = -compute_grad(inputs=xyz_list[0], output=v_bias).sum(0)

        keep_idx = self.get_keep_idx(atoms)
        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias.detach().cpu()
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.detach().numpy(), v_bias.detach().numpy()

    def get_bias(self, atoms):
        bias_type = self.pushing_params["bias_type"]
        if bias_type == "rmsd":
            results = self.rmsd_push(atoms)
        else:
            raise NotImplementedError

        return results

    def append_atoms(self, atoms):
        self.old_atoms.append(atoms)
        self.steps_from_old.append(0)

        max_ref = self.pushing_params.get("max_ref_structures")
        if max_ref is None:
            max_ref = 10

        if len(self.old_atoms) >= max_ref:
            self.old_atoms = self.old_atoms[-max_ref:]
            self.steps_from_old = self.steps_from_old[-max_ref:]

    def calculate(
        self,
        atoms,
        properties=["energy", "forces"],
        system_changes=all_changes,
        add_steps=True,
    ):
        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        # Add metadynamics energy and forces

        f_bias, _ = self.get_bias(atoms)

        self.results["forces"] += f_bias
        self.results["f_bias"] = f_bias

        if add_steps:
            for i, step in enumerate(self.steps_from_old):
                self.steps_from_old[i] = step + 1


class BatchNeuralMetadynamics(NeuralMetadynamics):
    def __init__(
        self,
        model,
        pushing_params,
        old_atoms=None,
        device="cpu",
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        NeuralMetadynamics.__init__(
            self,
            model=model,
            pushing_params=pushing_params,
            old_atoms=old_atoms,
            device=device,
            en_key=en_key,
            directed=directed,
            **kwargs,
        )

        self.query_nxyz = None
        self.mol_idx = None

    def rmsd_prelims(self, atoms):
        # f_damp is the turn-on on timescale, measured in
        # number of steps. From https://pubs.acs.org/doi/pdf/
        # 10.1021/acs.jctc.9b00143

        kappa = self.pushing_params["kappa"]
        steps_from_old = torch.Tensor(self.steps_from_old)
        f_damp = 2 / (1 + torch.exp(-kappa * steps_from_old)) - 1

        # k_i depends on the number of atoms so must be done by batch
        # given in mHartree / atom in CREST paper

        k_i = self.pushing_params["k_i"] / 1000 * units.Hartree * atoms.num_atoms

        # given in Bohr^(-2) in CREST paper
        alpha_i = self.pushing_params["alpha_i"] / units.Bohr**2

        return k_i, alpha_i, f_damp

    def get_query_nxyz(self, keep_idx):
        if self.query_nxyz is not None:
            return self.query_nxyz

        query_nxyz = torch.stack(
            [
                torch.Tensor(old_atoms.get_nxyz())[keep_idx, :]
                for old_atoms in self.old_atoms
            ]
        )
        self.query_nxyz = query_nxyz

        return query_nxyz

    def append_atoms(self, atoms):
        super().append_atoms(atoms)
        self.query_nxyz = None

    def make_nxyz(self, atoms):
        keep_idx = self.get_keep_idx(atoms)
        ref_nxyz = torch.Tensor(atoms.get_nxyz())[keep_idx, :]
        query_nxyz = self.get_query_nxyz(keep_idx)

        return ref_nxyz, query_nxyz, keep_idx

    def get_mol_idx(self, atoms, keep_idx):
        if self.mol_idx is not None:
            assert self.mol_idx.max() + 1 == len(atoms.num_atoms)
            return self.mol_idx

        num_atoms = atoms.num_atoms
        counter = 0

        mol_idx = []

        for i, num in enumerate(num_atoms):
            mol_idx.append(torch.ones(num).long() * i)
            counter += num

        mol_idx = torch.cat(mol_idx)[keep_idx]
        self.mol_idx = mol_idx

        return mol_idx

    def get_num_atoms_tensor(self, mol_idx, atoms):
        num_atoms = torch.LongTensor(
            [(mol_idx == i).nonzero().shape[0] for i in range(len(atoms.num_atoms))]
        )

        return num_atoms

    def get_v_f_bias(self, rmsd, ref_xyz, k_i, alpha_i, f_damp):
        v_bias = (f_damp.reshape(-1, 1) * k_i * torch.exp(-alpha_i * rmsd**2)).sum()

        f_bias = -compute_grad(inputs=ref_xyz, output=v_bias)

        output = [v_bias.reshape(-1).detach().cpu(), f_bias.detach().cpu()]

        return output

    def rmsd_push(self, atoms):
        if not self.old_atoms:
            return np.zeros((len(atoms), 3)), np.zeros(len(atoms.num_atoms))

        k_i, alpha_i, f_damp = self.rmsd_prelims(atoms)

        ref_nxyz, query_nxyz, keep_idx = self.make_nxyz(atoms=atoms)
        mol_idx = self.get_mol_idx(atoms=atoms, keep_idx=keep_idx)
        num_atoms_tensor = self.get_num_atoms_tensor(mol_idx=mol_idx, atoms=atoms)

        # note - everything is done on CPU, which is much faster than GPU. E.g. for
        # 30 molecules in a batch, each around 70 atoms, it's 4 times faster to do
        # this on CPU than GPU

        rmsd, ref_xyz = batch_compute_distance(
            ref_nxyz=ref_nxyz,
            query_nxyz=query_nxyz,
            mol_idx=mol_idx,
            num_atoms_tensor=num_atoms_tensor,
            store_grad=True,
        )

        v_bias, f_bias = self.get_v_f_bias(
            rmsd=rmsd, ref_xyz=ref_xyz, k_i=k_i, alpha_i=alpha_i, f_damp=f_damp
        )

        final_f_bias = torch.zeros(len(atoms), 3)
        final_f_bias[keep_idx] = f_bias
        nan_idx = torch.bitwise_not(torch.isfinite(final_f_bias))
        final_f_bias[nan_idx] = 0

        return final_f_bias.numpy(), v_bias.numpy()


class NeuralGAMD(NeuralFF):
    """
    NeuralFF for Gaussian-accelerated molecular dynamics (GAMD)
    """

    def __init__(
        self,
        model,
        k_0,
        V_min,
        V_max,
        device=0,
        en_key="energy",
        directed=DEFAULT_DIRECTED,
        **kwargs,
    ):
        NeuralFF.__init__(
            self,
            model=model,
            device=device,
            en_key=en_key,
            directed=DEFAULT_DIRECTED,
            **kwargs,
        )
        self.V_min = V_min
        self.V_max = V_max

        self.k_0 = k_0
        self.k = self.k_0 / (self.V_max - self.V_min)

    def calculate(
        self, atoms, properties=["energy", "forces"], system_changes=all_changes
    ):
        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        old_en = self.results["energy"]
        if old_en < self.V_max:
            old_forces = self.results["forces"]
            f_bias = -self.k * (self.V_max - old_en) * old_forces

            self.results["forces"] += f_bias


class ProjVectorCentroid:
    """
    Collective variable class. Projection of a position vector onto a reference vector
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    vector: list of int
       List of the indices of atoms that define the vector on which the position vector is projected
    indices: list if int
       List of indices of the mol/fragment
    reference: list of int
       List of atomic indices that are used as reference for the position vector

    note: the position vector is calculated in the method get_value
    """

    def __init__(self, vector=[], indices=[], reference=[], device="cpu"):
        self.vector_inds = vector
        self.mol_inds = torch.LongTensor(indices)
        self.reference_inds = reference

    def get_value(self, positions):
        vector_pos = positions[self.vector_inds]
        vector = vector_pos[1] - vector_pos[0]
        vector = vector / torch.linalg.norm(vector)
        mol_pos = positions[self.mol_inds]
        reference_pos = positions[self.reference_inds]
        mol_centroid = mol_pos.mean(axis=0)  # mol center
        reference_centroid = reference_pos.mean(
            axis=0
        )  # centroid of the whole structure

        # position vector with respect to the structure centroid
        rel_mol_pos = mol_centroid - reference_centroid

        # projection
        cv = torch.dot(rel_mol_pos, vector)
        return cv


class ProjVectorPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.

    note: the position vector is calculated in the method get_value
    """

    def __init__(self, mol_inds=[], ring_inds=[]):
        self.mol_inds = torch.LongTensor(mol_inds)  # list of indices
        self.ring_inds = torch.LongTensor(ring_inds)  # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = []
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0)  # mol center
        ring_cm = self.ring_coors.mean(axis=0)  # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors)  # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1 / N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2 / N

        plane_vec = torch.cross(r1, r2)
        plane_vec = plane_vec / torch.linalg.norm(plane_vec)
        pos_vec = mol_cm - ring_cm

        cv = torch.dot(pos_vec, plane_vec)
        return cv


class ProjOrthoVectorsPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.

    note: the position vector is calculated in the method get_value
    """

    def __init__(self, mol_inds=[], ring_inds=[]):
        self.mol_inds = torch.LongTensor(mol_inds)  # list of indices
        self.ring_inds = torch.LongTensor(ring_inds)  # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = []
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0)  # mol center
        ring_cm = self.ring_coors.mean(axis=0)  # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors)  # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1 / N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2 / N

        # normalize r1 and r2
        r1 = r1 / torch.linalg.norm(r1)
        r2 = r2 / torch.linalg.norm(r2)
        # project position vector on r1 and r2
        pos_vec = mol_cm - ring_cm
        proj1 = torch.dot(pos_vec, r1)
        proj2 = torch.dot(pos_vec, r2)
        cv = proj1 + proj2
        return abs(cv)


class HarmonicRestraint:
    """Class to apply a harmonic restraint on a MD simulations
    Params
    ------
    cvdic (dict): Dictionary contains the information to define the collective variables
                  and the harmonic restraint.
    max_steps (int): maximum number of steps of the MD simulation
    device: device
    """

    def __init__(self, cvdic, max_steps, device="cpu"):
        self.cvs = []  # list of collective variables (CV) objects
        self.kappas = []  # list of constant values for every CV
        self.eq_values = []  # list equilibrium values for every CV
        self.steps = []  # list of lists with the steps of the MD simulation
        self.setup_contraint(cvdic, max_steps, device)

    def setup_contraint(self, cvdic, max_steps, device):
        """Initializes the collectiva variables objects
        Args:
            cvdic (dict): Dictionary contains the information to define the collective variables
                          and the harmonic restraint.
            max_steps (int): maximum number of steps of the MD simulation
            device: device
        """
        for cvname, val in cvdic.items():
            if val["type"].lower() == "proj_vec_plane":
                mol_inds = [i - 1 for i in val["mol"]]  # caution check type
                ring_inds = [i - 1 for i in val["ring"]]
                cv = ProjVectorPlane(mol_inds, ring_inds)
            elif val["type"].lower() == "proj_vec_ref":
                mol_inds = [i - 1 for i in val["mol"]]
                reference = [i - 1 for i in val["reference"]]
                vector = [i - 1 for i in val["vector"]]
                cv = ProjVectorCentroid(
                    vector, mol_inds, reference, device=device
                )  # z components
            elif val["type"].lower() == "proj_ortho_vectors_plane":
                mol_inds = [i - 1 for i in val["mol"]]  # caution check type
                ring_inds = [i - 1 for i in val["ring"]]
                cv = ProjOrthoVectorsPlane(mol_inds, ring_inds)  # z components
            else:
                print("Bad CV type")
                sys.exit(1)

            self.cvs.append(cv)
            steps, kappas, eq_values = self.create_time_dependec_arrays(
                val["restraint"], max_steps
            )
            self.kappas.append(kappas)
            self.steps.append(steps)
            self.eq_values.append(eq_values)

    def create_time_dependec_arrays(self, restraint_list, max_steps):
        """creates lists of steps, kappas and equilibrium values that will be used along
        the simulation to determine the value of kappa and equilibrium CV at each step

        Args:
            restraint_list (list of dicts): contains dictionaries with the desired values of
                                            kappa and CV at arbitrary step in the simulation
            max_steps (int): maximum number of steps of the simulation

        Returns:
            list: all steps e.g [1,2,3 .. max_steps]
            list: all kappas for every step, e.g [0.5, 0.5, 0.51, .. ] same size of steps
            list: all equilibrium values for every step, e.g. [1,1,1,3,3,3, ...], same size of steps
        """
        steps = []
        kappas = []
        eq_vals = []
        # in case the restraint does not start at 0
        templist = list(range(0, restraint_list[0]["step"]))
        steps += templist
        kappas += [0 for _ in templist]
        eq_vals += [0 for _ in templist]

        for n, rd in enumerate(restraint_list[1:]):
            # rd and n are out of phase by 1, when n = 0, rd points to index 1
            templist = list(range(restraint_list[n]["step"], rd["step"]))
            steps += templist
            kappas += [restraint_list[n]["kappa"] for _ in templist]
            dcv = rd["eq_val"] - restraint_list[n]["eq_val"]
            cvstep = dcv / len(templist)  # get step increase
            eq_vals += [
                restraint_list[n]["eq_val"] + cvstep * tind
                for tind, _ in enumerate(templist)
            ]

        # in case the last step is lesser than the max_step
        templist = list(range(restraint_list[-1]["step"], max_steps))
        steps += templist
        kappas += [restraint_list[-1]["kappa"] for _ in templist]
        eq_vals += [restraint_list[-1]["eq_val"] for _ in templist]

        return steps, kappas, eq_vals

    def get_energy(self, positions, step):
        """Calculates the retraint energy of an arbritrary state of the system

        Args:
            positions (torch.tensor): atomic positions
            step (int): current step

        Returns:
            float: energy
        """
        tot_energy = 0
        for n, cv in enumerate(self.cvs):
            kappa = self.kappas[n][step]
            eq_val = self.eq_values[n][step]
            cv_value = cv.get_value(positions)
            energy = 0.5 * kappa * (cv_value - eq_val) * (cv_value - eq_val)
            tot_energy += energy
        return tot_energy

    def get_bias(self, positions, step):
        """Calculates the bias energy and force

        Args:
            positions (torch.tensor): atomic positions
            step (int): current step

        Returns:
            float: forces
            float: energy
        """
        energy = self.get_energy(positions, step)
        forces = grad(energy, positions)[0]
        return forces, energy


class NeuralRestraint(Calculator):
    """ASE calculator to run Neural restraint MD simulations"""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model,
        device="cpu",
        en_key="energy",
        properties=["energy", "forces"],
        cv={},
        max_steps=0,
        **kwargs,
    ):
        """Creates a NeuralFF calculator.nff/io/ase.py

        Args:
            model (TYPE): Description
            device (str): device on which the calculations will be performed
            properties (list of str): 'energy', 'forces' or both and also stress for only schnet and painn
            **kwargs: Description
            model (one of nff.nn.models)
        """

        Calculator.__init__(self, **kwargs)
        self.model = model
        self.model.eval()
        self.device = device
        self.to(device)
        self.en_key = en_key
        self.step = -1
        self.max_steps = max_steps
        self.properties = properties
        self.hr = HarmonicRestraint(cv, max_steps, device)

    def to(self, device):
        self.device = device
        self.model.to(device)

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces"],
        system_changes=all_changes,
    ):
        """Calculates the desired properties for the given AtomsBatch.

        Args:
            atoms (AtomsBatch): custom Atoms subclass that contains implementation
                of neighbor lists, batching and so on. Avoids the use of the Dataset
                to calculate using the models created.
            system_changes (default from ase)
        """

        # print("calculating ...")
        self.step += 1
        # print("step ", self.step, self.step*0.0005)
        if not any([isinstance(self.model, i) for i in UNDIRECTED]):
            check_directed(self.model, atoms)

        # for backwards compatability
        if getattr(self, "properties", None) is None:
            self.properties = properties

        Calculator.calculate(self, atoms, self.properties, system_changes)

        # run model
        batch = batch_to(atoms.get_batch(), self.device)

        # add keys so that the readout function can calculate these properties
        grad_key = self.en_key + "_grad"
        batch[self.en_key] = []
        batch[grad_key] = []

        kwargs = {}
        requires_stress = "stress" in self.properties
        if requires_stress:
            kwargs["requires_stress"] = True
        prediction = self.model(batch, **kwargs)

        # change energy and force to kcal/mol
        energy = prediction[self.en_key] * (1 / const.EV_TO_KCAL_MOL)
        energy_grad = prediction[grad_key] * (1 / const.EV_TO_KCAL_MOL)

        # bias ------------------
        bias_forces, bias_energy = self.hr.get_bias(
            torch.tensor(atoms.get_positions(), requires_grad=True, device=self.device),
            self.step,
        )
        energy_grad += bias_forces

        # change energy and force to numpy array
        energy_grad = energy_grad.detach().cpu().numpy()
        energy = energy.detach().cpu().numpy()

        self.results = {"energy": energy.reshape(-1)}

        if "forces" in self.properties:
            self.results["forces"] = -energy_grad.reshape(-1, 3)
        if requires_stress:
            stress = prediction["stress_volume"].detach().cpu().numpy() * (
                1 / const.EV_TO_KCAL_MOL
            )
            self.results["stress"] = stress * (1 / atoms.get_volume())

        with open("colvar", "a") as f:
            f.write("{} ".format(self.step * 0.5))
            # ARREGLAR, SI YA ESTA CALCULADO PARA QUE RECALCULAR LA CVS
            for cv in self.hr.cvs:
                curr_cv_val = float(
                    cv.get_value(
                        torch.tensor(atoms.get_positions(), device=self.device)
                    )
                )
                f.write(" {:.6f} ".format(curr_cv_val))
            f.write("{:.6f} \n".format(float(bias_energy)))

    @classmethod
    def from_file(cls, model_path, device="cuda", **kwargs):
        model = load_model(model_path)
        out = cls(model=model, device=device, **kwargs)
        return out
