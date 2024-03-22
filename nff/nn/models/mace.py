####################################################################################################
# A wrapper for MACE models
# Authors: Hoje Chun, Juno Nam, Alex Hoffman
# Distributed under the MIT license.
# See LICENSE for more info
####################################################################################################

import os
import urllib
from typing import Union

import torch
from mace.calculators.mace import get_model_dtype
from mace.data.atomic_data import AtomicData, AtomicNumberTable, torch_geometric
from mace.data.utils import Configuration
from mace.modules.models import MACE, ScaleShiftMACE
from mace.tools import torch_tools

# get the path to NFF models dir, which is the parent directory of this file
module_dir = os.path.abspath(
    os.path.join(os.path.abspath(__file__), "..", "..", "..", "..", "models")
)
print(module_dir)
LOCAL_MODEL_PATH = os.path.join(
    module_dir, "foundation_models/mace/2023-12-03-mace-mp.model"
)

MACE_URLS = dict(
    small="http://tinyurl.com/46jrkm3v",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
    medium="http://tinyurl.com/5yyxdm76",  # 2023-12-03-mace-128-L1_epoch-199.model
    large="http://tinyurl.com/5f5yavf3",  # MACE_MPtrj_2022.9.model
)


def get_mace_mp_model(model: str = None) -> str:
    """Get the default MACE MP model. Replicated from the MACE codebase,
    Copyright (c) 2022 ACEsuit/mace and licensed under the MIT license.

    Args:
        model (str, optional): MACE_MP model that you want to get.
            Defaults to None. Can be "small", "medium", "large", or a URL.

    Raises:
        RuntimeError: raised if the model download fails and no local model is found

    Returns:
        str: path to the model
    """
    if model in (None, "medium") and os.path.isfile(LOCAL_MODEL_PATH):
        model = LOCAL_MODEL_PATH
        print(
            f"Using local medium Materials Project MACE model for MACECalculator {model}"
        )
    elif model in (None, "small", "medium", "large") or str(model).startswith("https:"):
        try:
            checkpoint_url = (
                MACE_URLS.get(model, MACE_URLS["medium"])
                if model in (None, "small", "medium", "large")
                else model
            )
            cache_dir = os.path.expanduser("~/.cache/mace")
            checkpoint_url_name = "".join(
                c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
            )
            cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
            if not os.path.isfile(cached_model_path):
                os.makedirs(cache_dir, exist_ok=True)
                # download and save to disk
                print(f"Downloading MACE model from {checkpoint_url!r}")
                urllib.request.urlretrieve(checkpoint_url, cached_model_path)
                print(f"Cached MACE model to {cached_model_path}")
            model = cached_model_path
            msg = f"Loading Materials Project MACE with {model}"
            print(msg)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed and no local model found"
            ) from exc

    return model


class NffScaleMACE(ScaleShiftMACE):
    """Wrapper for the ScaleShiftMACE model that allows for direct
    forward passes in NFF
    """

    def __init__(
        self,
        mace_model: ScaleShiftMACE,
        default_dtype: str = "",
        training: bool = False,
        **kwargs,
    ):
        """Initialize the NffScaleMACE model

        Args:
            mace_model (ScaleShiftMACE): MACE model that you want to wrap
            default_dtype (str, optional): default dtype for the model, either
                "float64" or "float32". Defaults to "", which will use the dtype
                from the input MACE model.
            training (bool, optional): training mode for the model. Defaults to False.
                Needs to be true if you want to fine-tune.
            kwargs: additional keyword arguments for the super().init() function from MACE
        """
        atomic_inter_scale = mace_model.scale_shift.scale
        atomic_inter_shift = mace_model.scale_shift.shift
        super().__init__(atomic_inter_scale, atomic_inter_shift, **kwargs)
        self.z_table = AtomicNumberTable([int(z) for z in mace_model.atomic_numbers])

        # set the model to training mode if necessary because the
        # MACE model is not set to training mode by default and takes
        # the training argument in the forward pass
        self.training = training

        model_dtype = get_model_dtype(mace_model)
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
                self.double()
            elif default_dtype == "float32":
                self.float()
        torch_tools.set_default_dtype(default_dtype)

    def forward(
        self,
        batch: dict,
        training: bool = False,  # retained to keep the same args as the original MACE model forward
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> dict:  # noqa: W0221
        """Forward pass through the model

        Args:
            batch (dict): dictionary of Geom properties (energy, energy_grad, stress, etc.)
            kwargs: additional keyword arguments for the super().forward() function from MACE

        Returns:
            dict: dict of output from the forward pass
        """
        data = self.convert_batch_to_data(batch)
        output = super().forward(
            data,
            training=self.training,  # set the training mode to the value of the wrapper
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
        )
        output = self.add_engrad_to_output(output)
        return output

    def convert_batch_to_data(self, batch: dict) -> torch_geometric.data.Data:
        """Convert Batch object to a torch geometric data object
        that is used in the MACE model

        Args:
            batch (dict): a batch object that contains the properties. This
                function will also work with Trainers that use the batches
                from Dataloaders directly.

        Raises:
            ValueError: raised if no cell or lattice is found in the batch

        Returns:
            torch_geometric.data.Data: torch geometric data object
        """
        if isinstance(batch, dict):
            props = batch
        else:
            raise ValueError("Batch must be a dictionary")
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
    def get_init_args_from_model(cls, model: Union[ScaleShiftMACE, MACE]) -> dict:
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
    def from_file(
        cls, path: str, map_location: str = "cpu", default_dtype: str = ""
    ) -> "NffScaleMACE":
        """Build a NffScaleMACE from a saved file."""
        state = torch.load(path, map_location=map_location)
        # extract initialization parameters from the state
        init_params = NffScaleMACE.get_init_args_from_model(state)
        return NffScaleMACE(
            mace_model=state, default_dtype=default_dtype, **init_params
        )

    @classmethod
    def load(
        cls, model: str = "medium", map_location: str = "cpu", default_dtype: str = ""
    ) -> "NffScaleMACE":
        """Load a MACE model from a file or URL"""
        mace_model = get_mace_mp_model(model)
        return cls.from_file(
            mace_model, map_location=map_location, default_dtype=default_dtype
        )
