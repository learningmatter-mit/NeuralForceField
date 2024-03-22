"""
Various functions to interface with MACE objects.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import ase
import ase.data
import numpy as np
import torch
from nff.io.ase import AtomsBatch

from mace.data.utils import Configuration

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


def mace_config_from_atoms_batch(
    atoms_batch: AtomsBatch,
    energy_key="energy",
    energy_grad_key="energy_grad",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """The function `mace_config_from_atoms_batch` converts a batch of atoms in NFF format to a configuration in
    MACE format.

    Parameters
    ----------
    atoms_batch : AtomsBatch
    The `atoms_batch` parameter is an object of the `AtomsBatch` class. It contains the data that needs to be
    converted to the `config` format.
    energy_key : str
    The `energy_key` parameter is a string that represents the key for the energy in the `atoms_batch` object.
    energy_grad_key : str
    The `energy_grad_key` parameter is a string that represents the key for the energy gradient in the `atoms_batch` object.
    stress_key : str
    The `stress_key` parameter is a string that represents the key for the stress in the `atoms_batch` object.
    virials_key : str
    The `virials_key` parameter is a string that represents the key for the virials in the `atoms_batch` object.
    dipole_key : str
    The `dipole_key` parameter is a string that represents the key for the dipole in the `atoms_batch` object.
    charges_key : str
    The `charges_key` parameter is a string that represents the key for the charges in the `atoms_batch` object.
    config_type_weights : Dict[str, float]
    The `config_type_weights` parameter is a dictionary that represents the weights for the different types of
    configurations in the `atoms_batch` object.

    Returns
    -------
    a `config` object of type `Configuration`."""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    atoms_batch.convert_props_units("eV")  # convert units to eV
    # print(f"current units: {atoms_batch.props['units']}")
    props = atoms_batch.props.copy()

    energy = props.get(energy_key, None)  # eV
    if energy is not None and len(energy) == 1:
        energy = energy[0]
    energy_grad = props.get(energy_grad_key, None)  # eV / Ang
    forces = (
        -torch.stack(energy_grad) if isinstance(energy_grad, list) else -energy_grad
    )  # eV / Ang
    stress = props.get(stress_key, None)  # eV / Ang
    virials = props.get(virials_key, None)
    dipole = props.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found

    charges = props.get(charges_key, np.zeros(len(atoms_batch)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms_batch.symbols]
    )
    pbc = tuple(atoms_batch.get_pbc())
    cell = np.array(atoms_batch.get_cell())
    config_type = props.get("config_type", "Default")
    weight = props.get("config_weight", 1.0) * config_type_weights.get(config_type, 1.0)
    energy_weight = props.get("config_energy_weight", 1.0)
    forces_weight = props.get("config_forces_weight", 1.0)
    stress_weight = props.get("config_stress_weight", 1.0)
    virials_weight = props.get("config_virials_weight", 1.0)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms_batch.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms_batch.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )
