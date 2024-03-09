from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Union

import torch
from chgnet.data.dataset import collate_graphs
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from chgnet.graph.crystalgraph import datatype
from chgnet.model import CHGNet
from chgnet.model.composition_model import AtomRef
from chgnet.model.encoders import AngleEncoder, AtomEmbedding, BondEncoder
from chgnet.model.functions import MLP, GatedMLP, find_normalization
from chgnet.model.layers import (
    AngleUpdate,
    AtomConv,
    BondConv,
    GraphAttentionReadOut,
    GraphPooling,
)
from chgnet.utils import cuda_devices_sorted_by_free_mem
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch import Tensor, nn

from nff.io.ase import AtomsBatch
from nff.io.chgnet import convert_data_batch
from nff.nn.modules.chgnet.data.dataset import StructureData
from nff.utils.misc import cat_props

if TYPE_CHECKING:
    from chgnet import PredTask

module_dir = os.path.dirname(os.path.abspath(__file__))


class CHGNetNFF(CHGNet):
    """Wrapper class for CHGNet model."""

    def __init__(
        self, *args, units: str = "eV", key_mappings=None, device="cpu", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.units = units
        self.device = device

        if not key_mappings:
            # map from CHGNet keys to NFF keys
            self.key_mappings = {
                "e": "energy",
                "f": "energy_grad",
                "s": "stress",
                "m": "magmom",
                "atoms_per_graph": "num_atoms",
            }
            self.negate_keys = ("f",)

    def forward(self, data_batch: Dict, **kwargs):
        """
        Convert data_batch to CHGNet format and run forward pass.

        Parameters
        ----------
        data_batch : Dict
            A dictionary of properties for each structure in the batch.
            Basically the props in NFF Dataset
            Example:
                props = {
                    'nxyz': [np.array([[1, 0, 0, 0], [1, 1.1, 0, 0]]),
                             np.array([[1, 3, 0, 0], [1, 1.1, 5, 0]])],
                    'lattice': [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])],
                    'num_atoms': [2, 2],
                }

        Returns
        -------
        output : Dict
            A dictionary of predicted properties for each structure in the batch.
            Example:
                props = {
                    'energy': [1, 1.2],
                    'energy_grad': [np.array([[0, 0, 0], [0.1, 0.2, 0.3]]),
                                      np.array([[0, 0, 0], [0.1, 0.2, 0.3]])],
                }
        """
        chgnet_data_batch = convert_data_batch(data_batch)
        graphs, targets = collate_graphs(chgnet_data_batch)

        graphs = [graph.to(self.device) for graph in graphs]

        output = super().forward(graphs, task="ef")

        # convert to NFF keys and negate energy_grad
        output = cat_props(
            {self.key_mappings[k]: self.negate_value(k, v) for k, v in output.items()}
        )

        return output

    def negate_value(self, key: str, value: Union[list, Tensor]) -> Union[list, Tensor]:
        if key in self.negate_keys:
            if isinstance(value, list):
                return [-x for x in value]
            return -value
        return value

    @classmethod
    def from_dict(cls, dict, **kwargs):
        """Build a CHGNetNFF from a saved dictionary."""
        chgnet = CHGNetNFF(**dict["model_args"], **kwargs)
        chgnet.load_state_dict(dict["state_dict"])
        return chgnet

    @classmethod
    def from_file(cls, path, **kwargs):
        """Build a CHGNetNFF from a saved file."""
        state = torch.load(path, map_location=torch.device("cpu"))
        return CHGNetNFF.from_dict(state["model"], **kwargs)

    @classmethod
    def load(cls, model_name="0.3.0", **kwargs):
        """Load pretrained CHGNetNFF model.

        Args:
            model_name (str, optional): Defaults to "0.3.0".

        Raises:
            ValueError: On unknown model_name.
        """
        checkpoint_path = {
            "0.3.0": "../pretrained/chgnet/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
            "0.2.0": "../pretrained/chgnet/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        }.get(model_name)

        if checkpoint_path is None:
            raise ValueError(f"Unknown {model_name=}")

        return cls.from_file(
            os.path.join(module_dir, checkpoint_path),
            mlp_out_bias=model_name == "0.2.0",
            version=model_name,
            **kwargs,
        )


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing.

    Attributes:
        atomic_numbers (Tensor): atomic numbers vector
            [num_batch_atoms]
        bond_bases_ag (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_ag, num_radial]
        bond_bases_bg (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_bg, num_radial]
        angle_bases (Tensor): angle bases vector
            [num_batch_angles, num_angular]
        batched_atom_graph (Tensor) : batched atom graph adjacency list
            [num_batch_bonds, 2]
        batched_bond_graph (Tensor) : bond graph adjacency list
            [num_batch_angles, 3]
        atom_owners (Tensor): graph indices for each atom, used aggregate batched
            graph back to single graph
            [num_batch_atoms]
        directed2undirected (Tensor): the utility tensor used to quickly
            map directed edges to undirected edges in graph
            [num_directed]
        atom_positions (list[Tensor]): cartesian coordinates of the atoms
            from structures
            [[num_atoms_1, 3], [num_atoms_2, 3], ...]
        strains (list[Tensor]): a list of strains that's initialized to be zeros
            [[3, 3], [3, 3], ...]
        volumes (Tensor): the volume of each structure in the batch
            [batch_size]
    """

    atomic_numbers: Tensor
    bond_bases_ag: Tensor
    bond_bases_bg: Tensor
    angle_bases: Tensor
    batched_atom_graph: Tensor
    batched_bond_graph: Tensor
    atom_owners: Tensor
    directed2undirected: Tensor
    atom_positions: Sequence[Tensor]
    strains: Sequence[Tensor]
    volumes: Sequence[Tensor]

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        bond_basis_expansion: nn.Module,
        angle_basis_expansion: nn.Module,
        compute_stress: bool = False,
    ) -> BatchedGraph:
        """Featurize and assemble a list of graphs.

        Args:
            graphs (list[Tensor]): a list of CrystalGraphs
            bond_basis_expansion (nn.Module): bond basis expansion layer in CHGNet
            angle_basis_expansion (nn.Module): angle basis expansion layer in CHGNet
            compute_stress (bool): whether to compute stress. Default = False

        Returns:
            BatchedGraph: assembled graphs ready for batched CHGNet forward pass
        """
        atomic_numbers, atom_positions = [], []
        strains, volumes = [], []
        bond_bases_ag, bond_bases_bg, angle_bases = [], [], []
        batched_atom_graph, batched_bond_graph = [], []
        directed2undirected = []
        atom_owners = []
        atom_offset_idx = 0
        n_undirected = 0

        for graph_idx, graph in enumerate(graphs):
            # Atoms
            n_atom = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number)

            # Lattice
            if compute_stress:
                strain = graph.lattice.new_zeros([3, 3], requires_grad=True)
                lattice = graph.lattice @ (
                    torch.eye(3, dtype=datatype).to(strain.device) + strain
                )
            else:
                strain = None
                lattice = graph.lattice
            volumes.append(torch.dot(lattice[0], torch.cross(lattice[1], lattice[2])))
            strains.append(strain)

            # Bonds
            atom_cart_coords = graph.atom_frac_coord @ lattice
            bond_basis_ag, bond_basis_bg, bond_vectors = bond_basis_expansion(
                center=atom_cart_coords[graph.atom_graph[:, 0]],
                neighbor=atom_cart_coords[graph.atom_graph[:, 1]],
                undirected2directed=graph.undirected2directed,
                image=graph.neighbor_image,
                lattice=lattice,
            )
            atom_positions.append(atom_cart_coords)
            bond_bases_ag.append(bond_basis_ag)
            bond_bases_bg.append(bond_basis_bg)

            # Indexes
            batched_atom_graph.append(graph.atom_graph + atom_offset_idx)
            directed2undirected.append(graph.directed2undirected + n_undirected)

            # Angles
            # Here we use directed edges to calculate angles, and
            # keep only the undirected graph index in the bond_graph,
            # So the number of columns in bond_graph reduce from 5 to 3
            if len(graph.bond_graph) != 0:
                bond_vecs_i = torch.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 2]
                )
                bond_vecs_j = torch.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 4]
                )
                angle_basis = angle_basis_expansion(bond_vecs_i, bond_vecs_j)
                angle_bases.append(angle_basis)

                bond_graph = graph.bond_graph.new_zeros([graph.bond_graph.shape[0], 3])
                bond_graph[:, 0] = graph.bond_graph[:, 0] + atom_offset_idx
                bond_graph[:, 1] = graph.bond_graph[:, 1] + n_undirected
                bond_graph[:, 2] = graph.bond_graph[:, 3] + n_undirected
                batched_bond_graph.append(bond_graph)

            atom_owners.append(torch.ones(n_atom, requires_grad=False) * graph_idx)
            atom_offset_idx += n_atom
            n_undirected += len(bond_basis_ag)

        # Make Torch Tensors
        atomic_numbers = torch.cat(atomic_numbers, dim=0)
        bond_bases_ag = torch.cat(bond_bases_ag, dim=0)
        bond_bases_bg = torch.cat(bond_bases_bg, dim=0)
        angle_bases = (
            torch.cat(angle_bases, dim=0) if len(angle_bases) != 0 else torch.tensor([])
        )
        batched_atom_graph = torch.cat(batched_atom_graph, dim=0)
        if batched_bond_graph != []:
            batched_bond_graph = torch.cat(batched_bond_graph, dim=0)
        else:  # when bond graph is empty or disabled
            batched_bond_graph = torch.tensor([])
        atom_owners = (
            torch.cat(atom_owners, dim=0).type(torch.int32).to(atomic_numbers.device)
        )
        directed2undirected = torch.cat(directed2undirected, dim=0)
        volumes = torch.tensor(volumes, dtype=datatype, device=atomic_numbers.device)

        return cls(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,
            strains=strains,
            volumes=volumes,
        )
