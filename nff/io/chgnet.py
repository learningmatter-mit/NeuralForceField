"""Convert NFF Dataset to CHGNet StructureData"""

from typing import Dict, List, TYPE_CHECKING
import functools
import random
import torch
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from nff.data import Dataset
from nff.io import AtomsBatch
from nff.utils.cuda import batch_detach, detach
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from collections.abc import Sequence

datatype = torch.float32

class StructureData(Dataset):
    """A simple torch Dataset of structures."""

    def __init__(
        self,
        structures: list[Structure],
        energies: list[float],
        forces: list[Sequence[Sequence[float]]],
        stresses: list[Sequence[Sequence[float]]] | None = None,
        magmoms: list[Sequence[Sequence[float]]] | None = None,
        structure_ids: list[str] | None = None,
        graph_converter: CrystalGraphConverter | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            structures (list[dict]): pymatgen Structure objects.
            energies (list[float]): [data_size, 1]
            forces (list[list[float]]): [data_size, n_atoms, 3]
            stresses (list[list[float]], optional): [data_size, 3, 3]
            magmoms (list[list[float]], optional): [data_size, n_atoms, 1]
            structure_ids (list[str], optional): a list of ids to track the structures
            graph_converter (CrystalGraphConverter, optional): Converts the structures
                to graphs. If None, it will be set to CHGNet 0.3.0 converter
                with AtomGraph cutoff = 6A.

        Raises:
            RuntimeError: if the length of structures and labels (energies, forces,
                stresses, magmoms) are not equal.
        """
        for idx, struct in enumerate(structures):
            if not isinstance(struct, Structure):
                raise ValueError(f"{idx} is not a pymatgen Structure object: {struct}")
        for name in "energies forces stresses magmoms structure_ids".split():
            labels = locals()[name]
            if labels is not None and len(labels) != len(structures):
                raise RuntimeError(
                    f"Inconsistent number of structures and labels: "
                    f"{len(structures)=}, len({name})={len(labels)}"
                )
        self.structures = structures
        self.energies = energies
        self.forces = forces
        self.stresses = stresses
        self.magmoms = magmoms
        self.structure_ids = structure_ids
        self.keys = np.arange(len(structures))
        random.shuffle(self.keys)
        self.graph_converter = graph_converter or CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3
        )
        self.failed_idx: list[int] = []
        self.failed_graph_id: dict[str, str] = {}

    def __len__(self) -> int:
        """Get the number of structures in this dataset."""
        return len(self.keys)

    @functools.cache  # Cache loaded structures
    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict]:
        """Get one graph for a structure in this dataset.

        Args:
            idx (int): Index of the structure

        Returns:
            crystal_graph (CrystalGraph): graph of the crystal structure
            targets (dict): list of targets. i.e. energy, force, stress
        """
        if idx not in self.failed_idx:
            graph_id = self.keys[idx]
            try:
                struct = self.structures[graph_id]
                if self.structure_ids is not None:
                    mp_id = self.structure_ids[graph_id]
                else:
                    mp_id = graph_id
                crystal_graph = self.graph_converter(
                    struct, graph_id=graph_id, mp_id=mp_id
                )
                targets = {
                    "e": torch.tensor(self.energies[graph_id], dtype=datatype),
                    "f": torch.tensor(self.forces[graph_id], dtype=datatype),
                }
                if self.stresses is not None:
                    # Convert VASP stress
                    targets["s"] = torch.tensor(
                        self.stresses[graph_id], dtype=datatype
                    ) * (-0.1)
                if self.magmoms is not None:
                    mag = self.magmoms[graph_id]
                    # use absolute value for magnetic moments
                    if mag is None:
                        targets["m"] = None
                    else:
                        targets["m"] = torch.abs(torch.tensor(mag, dtype=datatype))

                return crystal_graph, targets

            # Omit structures with isolated atoms. Return another randomly selected
            # structure
            except Exception:
                struct = self.structures[graph_id]
                self.failed_graph_id[graph_id] = struct.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)


def convert_nff_to_chgnet_structure_data(
    dataset: Dataset,
    cutoff: float = 5.0
):
    """
    Converts a dataset in NFF format to a dataset in CHGNet structure data format.

    Args:
        dataset (Dataset): An object of the Dataset class.
        cutoff (float, optional): Distance cutoff for constructing the neighbor list. Defaults to 5.0.
        shuffle (bool, optional): Whether the dataset should be shuffled. Defaults to True.

    Returns:
        StructureData: A CHGNet StructureData object.
    """
    dataset = dataset.copy()
    dataset.to_units("eV/atom")  # convert units to eV
    print(f"current units: {dataset.units}")
    atoms_batch_list = dataset.as_atoms_batches(cutoff=cutoff)
    pymatgen_structures = [AseAtomsAdaptor.get_structure(atoms_batch) for atoms_batch in atoms_batch_list]

    energies_per_atom = dataset.props["energy"]

    energy_grads = dataset.props["energy_grad"]
    forces = [-x for x in energy_grads] if isinstance(energy_grads, list) else -energy_grads
    stresses = dataset.props.get("stress", None)
    magmoms = dataset.props.get("magmoms", None)

    return StructureData(
        structures=pymatgen_structures,
        energies=energies_per_atom,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms
    )


def convert_chgnet_structure_targets_to_nff(
    structures: List[Structure],
    targets: List[Dict],
    stresses: bool = False,
    magmoms: bool = False,
) -> Dataset:
    """
    Converts a dataset in CHGNet structure JSON data format to a dataset in NFF format.

    Args:
        structures (List[Structure]): List of pymatgen structures.
        targets (List[Dict]): List of dictionaries containing the properties of each structure.
        stresses (bool, optional): Whether the dataset should include stresses. Defaults to False.
        magmoms (bool, optional): Whether the dataset should include magnetic moments. Defaults to False.

    Returns:
        Dataset: An NFF Dataset.
    """
    energies_per_atom = []
    energy_grad = []
    stresses_list = []
    magmoms_list = []
    for target in targets:
        energies_per_atom.append(target["e"])
        energy_grad.append(-target["f"])
        if stresses:
            stresses_list.append(target["s"])
        if magmoms:
            magmoms_list.append(target["m"])

    lattice = []
    num_atoms = []  # TODO: check if this is correct
    nxyz = []
    units = ["eV/atom" for _ in range(len(structures))]
    formula = []
    for structure in structures:
        atoms = structure.to_ase_atoms()
        lattice.append(atoms.cell.tolist())
        num_atoms.append(len(atoms))
        nxyz.append([torch.cat([torch.tensor([atom.number]), torch.tensor(atom.position)]).tolist() for atom in atoms])
        formula.append(atoms.get_chemical_formula())

    concated_batch = {
        "nxyz": nxyz,
        "lattice": lattice,
        "num_atoms": num_atoms,
        "energy": energies_per_atom,
        "energy_grad": energy_grad,
        "formula": formula,
        "units": units,
    }
    if stresses:
        concated_batch["stress"] = stresses_list
    if magmoms:
        concated_batch["magmoms"] = magmoms_list
    return Dataset(concated_batch, units=units[0])


def convert_chgnet_structure_data_to_nff(
    structure_data: StructureData,
    cutoff: float = 6.0,
    shuffle: bool = False,
) -> Dataset:
    """
    Converts a dataset in CHGNet structure data format to a dataset in NFF format.

    Args:
        structure_data (StructureData): A CHGNet StructureData object.
        cutoff (float, optional): Distance cutoff for constructing the neighbor list. Defaults to 6.0.
        shuffle (bool, optional): Whether the dataset should be shuffled. Defaults to False.

    Returns:
        Dataset: An NFF Dataset.
    """
    pymatgen_structures = structure_data.structures
    energies_per_atom = structure_data.energies
    energy_grad = (
        [-x for x in structure_data.forces] if isinstance(structure_data.forces, list) else -structure_data.forces
    )
    stresses = structure_data.stresses
    magmoms = structure_data.magmoms
    lattice = []
    num_atoms = [structure.num_sites for structure in pymatgen_structures]  # TODO: check if this is correct
    nxyz = []
    units = ["eV/atom" for _ in range(len(pymatgen_structures))]
    formula = [structure.composition.formula for structure in pymatgen_structures]
    for structure in pymatgen_structures:
        lattice.append(structure.lattice.matrix)
        nxyz.append(
            [torch.cat([torch.tensor([atom.species.number]), torch.tensor(atom.coords)]).tolist() for atom in structure]
        )

    concated_batch = {
        "nxyz": nxyz,
        "lattice": lattice,
        "num_atoms": num_atoms,
        "energy": energies_per_atom,
        "energy_grad": energy_grad,
        "stress": stresses,
        "magmoms": magmoms,
        "formula": formula,
        "units": units,
    }
    return Dataset(concated_batch, units=units[0])


def convert_data_batch(
    data_batch: Dict,
    cutoff: float = 5.0,
    shuffle: bool = False,
):
    """
    Converts a dataset in NFF format to a dataset in CHGNet structure data format.

    Args:
        data_batch (Dict): Dictionary of properties for each structure in the batch.
            Example:
                props = {
                    'nxyz': [np.array([[1, 0, 0, 0], [1, 1.1, 0, 0]]),
                             np.array([[1, 3, 0, 0], [1, 1.1, 5, 0]])],
                    'lattice': [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])],
                    'num_atoms': [2, 2],
                }
        cutoff (float, optional): Distance cutoff for neighbor list construction. Defaults to 5.0.
        shuffle (bool, optional): Whether the dataset should be shuffled. Defaults to True.

    Returns:
        StructureData: A CHGNet StructureData object.
    """
    detached_batch = batch_detach(data_batch)
    nxyz = detached_batch["nxyz"]
    atoms_batch = AtomsBatch(
        nxyz[:, 0].long(),
        props=detached_batch,
        positions=nxyz[:, 1:],
        cell=(detached_batch["lattice"][0] if "lattice" in detached_batch else None),
        pbc="lattice" in detached_batch,
        cutoff=cutoff,
        dense_nbrs=False,
    )
    atoms_list = atoms_batch.get_list_atoms()

    pymatgen_structures = [AseAtomsAdaptor.get_structure(atoms_batch) for atoms_batch in atoms_list]

    energies = torch.atleast_1d(data_batch.get("energy"))
    if energies is not None and len(energies) > 0:
        energies_per_atom = energies
    else:
        # fake energies
        energies_per_atom = torch.Tensor([0.0] * len(pymatgen_structures))
    energy_grads = data_batch.get("energy_grad")

    num_atoms = detach(data_batch["num_atoms"]).tolist()
    stresses = data_batch.get("stress")
    magmoms = data_batch.get("magmoms")

    if energy_grads is not None and len(energy_grads) > 0:
        forces = [-x for x in energy_grads] if isinstance(energy_grads, list) else -energy_grads
    else:
        forces = None

    if forces is not None and len(forces) > 0:
        # organize forces per structure
        forces = torch.split(torch.atleast_2d(forces), num_atoms)
    else:
        # fake forces for NFF Calculator
        forces = [torch.zeros_like(torch.Tensor(atoms_batch.get_positions())) for atoms_batch in atoms_list]

    if stresses is not None and len(stresses) > 0:
        stresses = torch.split(torch.atleast_2d(stresses), num_atoms)
    if magmoms is not None and len(magmoms) > 0:
        magmoms = torch.split(torch.atleast_2d(magmoms), num_atoms)

    return StructureData(
        structures=pymatgen_structures,
        energies=energies_per_atom,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms
    )
