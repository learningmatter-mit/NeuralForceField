###########################################################################################
# Higher Order Real Clebsch Gordan (based on e3nn by Mario Geiger)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

###########################################################################################
# Tools for torch
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

###########################################################################################
# Statistics utilities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


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


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: torch.Tensor,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (compute_virials or compute_stress) and displacement is not None:
        # forces come for free
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths
