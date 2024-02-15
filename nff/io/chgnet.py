"""Convert NFF Dataset to CHGNet StructureData"""

import torch
from nff.data import Dataset
from nff.io import AtomsBatch
from pymatgen.io.ase import AseAtomsAdaptor

from chgnet.data.dataset import StructureData


def convert_nff_to_chgnet_structure_data(
    dataset: Dataset,
    cutoff: float = 5.0,
):
    '''The function `convert_nff_to_chgnet_structure_data` converts a dataset in NFF format to a dataset in
    CHGNet structure data format.
    
    Parameters
    ----------
    dataset : Dataset
        The `dataset` parameter is an object of the `Dataset` class. It contains the data that needs to be
    converted to the `chgnet_dataset` format.
    cutoff : float
        The `cutoff` parameter is a float value that represents the distance cutoff for constructing the
    neighbor list in the conversion process. It determines the maximum distance between atoms within
    which they are considered neighbors. Any atoms beyond this distance will not be included in the
    neighbor list.
    
    Returns
    -------
    a `chgnet_dataset` object of type `StructureData`.
    
    '''
    dataset = dataset.copy()
    dataset.to_units("eV")  # convert units to eV
    print(f"current units: {dataset.units}")
    atoms_batch_list = dataset.as_atoms_batches(cutoff=cutoff)
    pymatgen_structures = [AseAtomsAdaptor.get_structure(atoms_batch) for atoms_batch in atoms_batch_list]

    energies = dataset.props["energy"]
    energies_per_atoms = [energy / len(structure) for energy, structure in zip(energies, pymatgen_structures)]

    energy_grads = dataset.props["energy_grad"]
    forces = [-x for x in energy_grads] if isinstance(energy_grads, list) else -energy_grads
    stresses = dataset.props.get("stress", None)
    magmoms = dataset.props.get("magmoms", None)

    chgnet_dataset = StructureData(
        structures=pymatgen_structures,
        energies=energies_per_atoms,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
    )

    return chgnet_dataset

