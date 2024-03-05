"""Convert NFF Dataset to CHGNet StructureData"""

from typing import Dict

import torch
from chgnet.data.dataset import StructureData
from pymatgen.io.ase import AseAtomsAdaptor

from nff.data import Dataset
from nff.io import AtomsBatch
from nff.utils.cuda import batch_detach, batch_to, detach


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

def convert_data_batch(
    data_batch: Dict,
    cutoff: float = 5.0,
):
    '''Converts a dataset in NFF format to a dataset in
    CHGNet structure data format.
    
    Parameters
    ----------
    data_batch : Dict
    cutoff : float
        The `cutoff` parameter is a float value that represents the distance cutoff for constructing the
    neighbor list in the conversion process. It determines the maximum distance between atoms within
    which they are considered neighbors. Any atoms beyond this distance will not be included in the
    neighbor list.
    
    Returns
    -------
    a `chgnet_dataset` object of type `StructureData`.
    
    '''
    detached_batch = batch_detach(data_batch)
    nxyz = detached_batch["nxyz"]
    atoms_batch= AtomsBatch(
        nxyz[:, 0].long(),
        props=detached_batch,
        positions=nxyz[:, 1:],
        cell=detached_batch["lattice"][0]
            if "lattice" in detached_batch.keys()
            else None,
        pbc="lattice" in detached_batch.keys(),
        cutoff=cutoff,
        dense_nbrs=False,
    )
    atoms_list = atoms_batch.get_list_atoms()

    pymatgen_structures = [AseAtomsAdaptor.get_structure(atoms_batch) for atoms_batch in atoms_list]

    energies = data_batch["energy"]
    energies_per_atoms = [energy / len(structure) for energy, structure in zip(energies, pymatgen_structures)]

    energy_grads = data_batch["energy_grad"]
    forces = [-x for x in energy_grads] if isinstance(energy_grads, list) else -energy_grads
    num_atoms = detach(data_batch["num_atoms"]).tolist()

    stresses = data_batch.get("stress", None)
    magmoms = data_batch.get("magmoms", None)
    if forces is not None:
        forces = torch.split(forces, num_atoms)
    if stresses is not None:
        stresses = torch.split(stresses, num_atoms)
    if magmoms is not None:
        magmoms = torch.split(magmoms, num_atoms)

    chgnet_dataset = StructureData(
        structures=pymatgen_structures,
        energies=energies_per_atoms,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
    )

    return chgnet_dataset