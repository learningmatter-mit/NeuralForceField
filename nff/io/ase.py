import numpy as np
import torch
from ase import Atoms, units
from ase.neighborlist import neighbor_list

import nff.utils.constants as const
from nff.data.sparse import sparsify_array
from nff.nn.graphop import split_and_sum
from nff.nn.utils import clean_matrix, lattice_points_in_supercell, torch_nbr_list

DEFAULT_CUTOFF = 5.0
DEFAULT_DIRECTED = False
DEFAULT_SKIN = 1.0


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
        if target_unit not in ["kcal/mol", "eV", "atomic"]:
            raise NotImplementedError(f"Unit {target_unit} not implemented")

        curr_unit = self.props.get("units", "eV")

        if target_unit == curr_unit:
            return

        conversion_factor = const.conversion_factors.get((curr_unit, target_unit))
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
            atoms = Atoms(
                Z[i].tolist(),
                molecule_xyz.numpy(),
                cell=cells[i].numpy() if self.pbc.any() else None,
                pbc=self.pbc,
            )

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
