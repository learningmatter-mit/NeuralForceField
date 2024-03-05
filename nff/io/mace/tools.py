###########################################################################################
# Statistics utilities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch
from e3nn.io import CartesianTensor

# Imports for nff interface with MACE
import ase
import ase.data
from nff.io.ase import AtomsBatch
from mace.data.utils import Configuration

# Constants for nff-MACE interface
DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()


def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if (directory is not None) and (tag is not None):
        os.makedirs(name=directory, exist_ok=True)
        path = os.path.join(directory, tag + ".log")
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)

        logger.addHandler(fh)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


def get_optimizer(
    name: str,
    amsgrad: bool,
    learning_rate: float,
    weight_decay: float,
    parameters: Iterable[torch.Tensor],
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(
            parameters, lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
        )

    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay
        )

    raise RuntimeError(f"Unknown optimizer '{name}'")


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        logging.debug(f"Saving info: {self.path}")
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")


###########################################################################################
# Tools for torch
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

TensorDict = Dict[str, torch.Tensor]


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def tensor_dict_to_device(td: TensorDict, device: torch.device) -> TensorDict:
    return {k: v.to(device) if v is not None else None for k, v in td.items()}


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_device(device_str: str) -> torch.device:
    if "cuda" in device_str:
        assert torch.cuda.is_available(), "No CUDA device available!"
        if ":" in device_str:
            # Check if the desired device is available
            assert int(device_str.split(":")[-1]) < torch.cuda.device_count()
        logging.info(
            f"CUDA version: {torch.version.cuda}, CUDA device: {torch.cuda.current_device()}"
        )
        torch.cuda.init()
        return torch.device(device_str)
    if device_str == "mps":
        assert torch.backends.mps.is_available(), "No MPS backend is available!"
        logging.info("Using MPS GPU acceleration")
        return torch.device("mps")

    logging.info("Using CPU")
    return torch.device("cpu")


dtype_dict = {"float32": torch.float32, "float64": torch.float64}


def set_default_dtype(dtype: str) -> None:
    torch.set_default_dtype(dtype_dict[dtype])


def get_complex_default_dtype():
    default_dtype = torch.get_default_dtype()
    if default_dtype == torch.float64:
        return torch.complex128

    if default_dtype == torch.float32:
        return torch.complex64

    raise NotImplementedError


def spherical_to_cartesian(t: torch.Tensor):
    """
    Convert spherical notation to cartesian notation
    """
    stress_cart_tensor = CartesianTensor("ij=ji")
    stress_rtp = stress_cart_tensor.reduced_tensor_products()
    return stress_cart_tensor.to_cartesian(t, rtp=stress_rtp)


def cartesian_to_spherical(t: torch.Tensor):
    """
    Convert cartesian notation to spherical notation
    """
    stress_cart_tensor = CartesianTensor("ij=ji")
    stress_rtp = stress_cart_tensor.reduced_tensor_products()
    return stress_cart_tensor.to_cartesian(t, rtp=stress_rtp)


def voigt_to_matrix(t: torch.Tensor):
    """
    Convert voigt notation to matrix notation
    :param t: (6,) tensor or (3, 3) tensor or (9,) tensor
    :return: (3, 3) tensor
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )


def init_wandb(project: str, entity: str, name: str, config: dict):
    import wandb

    wandb.init(project=project, entity=entity, name=name, config=config)


###########################################################################################
# Tools for MACE to interface with NFF
# Authors: Xiaochen Du
###########################################################################################


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
    MACE format. Used for training MACE models with NFF data.

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
