####################################################################################################
# A wrapper for MACE models
# Authors: Hoje Chun, Juno Nam, Alex Hoffman
# Distributed under the MIT license.
# See LICENSE for more info
####################################################################################################

from mace.modules.models import MACE, ScaleShiftMACE
from mace.data.utils import Configuration
from mace.data.atomic_data import AtomicData, AtomicNumberTable, torch_geometric
from mace.tools.torch_geometric.batch import Batch
from mace.tools import torch_tools
from mace.calculators.mace import get_model_dtype
from typing import Union
import torch
import torch.nn as nn
from nff.io.ase import AtomsBatch

INIT_PROPS = [
    "r_max",
    "num_bessel",
    "num_polynomial_cutoff",
    "max_ell",
    "interaction_cls",
    "interaction_cls_first",
    "num_interactions",
    "num_elements",
    "hidden_irreps",
    "MLP_irreps",
    "atomic_energies",
    "avg_num_neighbors",
    "atomic_numbers",
    "correlation",
    "gate",
]


class NFFMACEWrapper(nn.Module):
    """Wrapper for the MACE model that allows for interfacing with NFF
    models. This wrapper converts AtomsBatch objects to the format
    required by the MACE model and adds energy gradients to the output
    from a forward pass that computes the forces."""

    def __init__(
        self, mace_model: Union[MACE, ScaleShiftMACE], default_dtype="", **kwargs
    ):
        """Initializes the NFFMACEWrapper

        Args:
            mace_model (Union[MACE, ScaleShiftMACE]): a MACE model, can be loaded
                from a file of a pre-trained model
            default_dtype (str, optional): datatype for the torch model. Defaults
                to "", which will match whatever is in the model file. Can also
                be "float32" or "float64".
        """
        super().__init__()
        self.mace_model = mace_model
        self.z_table = AtomicNumberTable(
            [int(z) for z in self.mace_model.atomic_numbers]
        )
        self.r_max = self.mace_model.r_max.item()
        model_dtype = get_model_dtype(self.mace_model)
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.mace_model = self.mace_model.double()
            elif default_dtype == "float32":
                self.mace_model = self.mace_model.float()
        torch_tools.set_default_dtype(default_dtype)

    def forward(self, batch, **kwargs):
        data = self.convert_atomsbatch_to_data(batch)
        output = self.mace_model.forward(data)
        output = self.add_engrad_to_output(output)
        return output

    def convert_atomsbatch_to_data(self, batch):
        if not isinstance(batch, dict):
            props = batch.get_batch()
        else:
            props = batch
        cum_idx_list = [0] + torch.cumsum(props["num_atoms"], 0).tolist()
        dataset = []

        for i in range(props.get("num_atoms").shape[0]):
            node_idx = torch.arange(cum_idx_list[i], cum_idx_list[i + 1])
            positions = props.get("nxyz")[node_idx, 1:].cpu().numpy()
            numbers = props.get("nxyz")[node_idx, 0].long().cpu().numpy()

            if "cell" in props.keys():
                cell = props["cell"][3 * i : 3 * i + 3].cpu().numpy()
            elif "lattice" in props.keys():
                cell = props["lattice"][3 * i : 3 * i + 3].cpu().numpy()
            else:
                raise ValueError("No cell or lattice found in batch")
            config = Configuration(
                atomic_numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=(True, True, True),
            )
            dataset.append(
                AtomicData.from_config(config, z_table=self.z_table, cutoff=self.r_max)
            )

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=False,
            drop_last=False,
        )
        data = next(iter(data_loader)).to(props["nxyz"].device)

        return data

    def add_engrad_to_output(self, output: dict) -> dict:
        """Adds an energy gradient to the output dictionary, which is
        computed from the force prediction

        Args:
            output (dict): dictionary of the output of a forward pass
                through the model

        Returns:
            dict: updated output dictionary with "energy_grad" key and value
        """
        output["energy_grad"] = -output["forces"]
        return output

    @classmethod
    def from_file(cls, path: str, **kwargs):
        """Build a NFFMACEWrapper from a saved file."""
        state = torch.load(path)
        # extract initialization parameters from the state
        return NFFMACEWrapper(mace_model=state, **kwargs)


class DirectNffScaleMACEWrapper(ScaleShiftMACE):
    """Wrapper for the ScaleShiftMACE model that allows for direct
    forward passes with an AtomsBatch object in NFF
    """

    def __init__(
        self, mace_model: Union[MACE, ScaleShiftMACE], default_dtype="", **kwargs
    ):
        atomic_inter_scale = mace_model.scale_shift.scale
        atomic_inter_shift = mace_model.scale_shift.shift
        print(kwargs)
        super().__init__(atomic_inter_scale, atomic_inter_shift, **kwargs)
        self.mace_model = mace_model
        self.z_table = AtomicNumberTable(
            [int(z) for z in self.mace_model.atomic_numbers]
        )
        model_dtype = get_model_dtype(self.mace_model)
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.mace_model = self.mace_model.double()
            elif default_dtype == "float32":
                self.mace_model = self.mace_model.float()
        torch_tools.set_default_dtype(default_dtype)

    def forward(self, batch, **kwargs):
        data = self.convert_atomsbatch_to_data(batch)
        breakpoint()
        output = super().forward(data, **kwargs)
        output = self.add_engrad_to_output(output)
        return output

    def convert_atomsbatch_to_data(self, batch) -> torch_geometric.data.Data:
        """Convert AtomsBatch object to a torch geometric data object
        that is used in the MACE model

        Args:
            batch (AtomsBatch): an AtomsBatch object, similar to ASE Atoms

        Raises:
            ValueError: raised if no cell or lattice is found in the AtomsBatch

        Returns:
            torch_geometric.data.Data: torch geometric data object
        """
        if isinstance(batch, AtomsBatch):
            props = batch.get_batch()
        elif isinstance(batch, dict):
            props = batch
        elif isinstance(batch, Batch):
            return batch
        else:
            print(type(batch))
            raise ValueError("Batch must be an AtomsBatch, Batch, or dictionary")
        cum_idx_list = [0] + torch.cumsum(props["num_atoms"], 0).tolist()
        dataset = []

        for i in range(props.get("num_atoms").shape[0]):
            node_idx = torch.arange(cum_idx_list[i], cum_idx_list[i + 1])
            positions = props.get("nxyz")[node_idx, 1:].cpu().numpy()
            numbers = props.get("nxyz")[node_idx, 0].long().cpu().numpy()

            if "cell" in props.keys():
                cell = props["cell"][3 * i : 3 * i + 3].cpu().numpy()
            elif "lattice" in props.keys():
                cell = props["lattice"][3 * i : 3 * i + 3].cpu().numpy()
            else:
                raise ValueError("No cell or lattice found in batch")
            config = Configuration(
                atomic_numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=(True, True, True),
            )
            if isinstance(self.r_max, float):
                r_max = self.r_max
            elif isinstance(self.r_max, torch.Tensor):
                r_max = self.r_max.item()
            dataset.append(
                AtomicData.from_config(config, z_table=self.z_table, cutoff=r_max)
            )

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=False,
            drop_last=False,
        )
        data = next(iter(data_loader)).to(props["nxyz"].device)

        return data

    def add_engrad_to_output(self, output: dict) -> dict:
        """Adds an energy gradient to the output dictionary, which is
        computed from the force prediction

        Args:
            output (dict): dictionary of the output of a forward pass
                through the model

        Returns:
            dict: updated output dictionary with "energy_grad" key and value
        """
        output["energy_grad"] = -output["forces"]
        return output

    @classmethod
    def get_init_args_from_model(self, model: Union[ScaleShiftMACE, MACE]) -> dict:
        """Get the initialization arguments from the model"""
        init_args = {
            "r_max": model.r_max.item(),
            "num_bessel": model.radial_embedding.out_dim,
            "num_polynomial_cutoff": model.radial_embedding.cutoff_fn.p.item(),
            "max_ell": model.spherical_harmonics.irreps_out.lmax,
            "interaction_cls": type(model.interactions[1]),
            "interaction_cls_first": type(model.interactions[0]),
            "num_interactions": model.num_interactions.item(),
            "num_elements": model.node_embedding.linear.irreps_in.dim,
            "hidden_irreps": model.interactions[0].hidden_irreps,
            "MLP_irreps": model.readouts[-1].hidden_irreps,
            "atomic_energies": model.atomic_energies_fn.atomic_energies,
            "avg_num_neighbors": model.interactions[0].avg_num_neighbors,
            "atomic_numbers": model.atomic_numbers.tolist(),
            "correlation": model.products[0]
            .symmetric_contractions.contractions[0]
            .correlation,
            "gate": model.readouts[-1].non_linearity.acts[0].f,
        }
        return init_args

    @classmethod
    def from_file(cls, path: str, map_location: str = "cpu", **kwargs):
        """Build a DirectNffScaleMACEWrapper from a saved file."""
        state = torch.load(path, map_location=map_location)
        # extract initialization parameters from the state
        init_params = DirectNffScaleMACEWrapper.get_init_args_from_model(state)
        return DirectNffScaleMACEWrapper(mace_model=state, **init_params)
