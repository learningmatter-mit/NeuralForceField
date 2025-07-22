from __future__ import annotations
from copy import deepcopy
from typing import Optional, Callable

from ase.calculators.calculator import Calculator, all_changes
from ase.symbols import Symbols
import numpy as np
import torch

from nff.io.ase_calcs import AtomsBatch



class Potential(torch.nn.Module):
    pass


class AsePotential(Potential):

    def __init__(self, calculator: Calculator, embedding_fun: Optional[Callable[[AtomsBatch], torch.Tensor]] = None) \
            -> None:
        super().__init__()
        self.calculator = calculator
        self.embedding_fun = embedding_fun

    def __call__(self, batch: dict, **kwargs):
        properties = ["energy"]
        if kwargs.get("requires_stress", False):
            properties.append("stress")
        if kwargs.get("requires_forces", False):
            properties.append("forces")
        if kwargs.get("requires_dipole", False):
            properties.append("dipole")
        if kwargs.get("requires_charges", False):
            properties.append("charges")
        if kwargs.get("requires_embedding", False):
            if self.embedding_fun is None:
                raise RuntimeError("Required embedding but no embedding function provided.")
            embedding = self.embedding_fun(batch)
        else:
            embedding = None

        nxyz = batch.get("nxyz")
        if nxyz is None:
            raise RuntimeError("Batch is missing 'nxyz' key.")
        pbc = batch.get("pbc")
        if pbc is not None:
            pbc = np.array(pbc, dtype=bool)
            cell = np.array(batch.get("cell")).reshape(3, 3)
        else:
            cell = None
        atoms_batch = AtomsBatch(
            symbols=Symbols(nxyz[:,0].detach().cpu().numpy()),
            positions=nxyz[:, 1:4].detach().cpu().numpy(),
            pbc=pbc,
            cell=cell,
            device=batch.get("device", "cpu")
        )

        self.calculator.calculate(atoms_batch, properties=properties, system_changes=all_changes)
        results = deepcopy(self.calculator.results)
        for key, value in results.items():
            if isinstance(value, str):
                continue
            if not hasattr(value, "__iter__"):
                results[key] = torch.tensor([value], device=atoms_batch.device)
            else:
                results[key] = torch.tensor(value, device=atoms_batch.device)
        if embedding is not None:
            results["embedding"] = embedding
        return results