####################################################################################################
# A wrapper for MACE models
# Authors: Hoje Chun, Juno Nam, Alex Hoffman
# Distributed under the MIT license.
# See LICENSE for more info
####################################################################################################

from __future__ import annotations

from typing import List, Union

import torch
from mace.calculators.mace import get_model_dtype
from mace.data.atomic_data import (AtomicData, AtomicNumberTable,
                                   torch_geometric)
from mace.data.utils import Configuration
from mace.modules.models import MACE, ScaleShiftMACE
from mace.modules.radial import BesselBasis, GaussianBasis
from mace.tools import torch_tools
# from mace.tools.torch_geometric.batch import Batch
from nff.io.mace import (get_atomic_number_table_from_zs,
                         get_init_kwargs_from_model, get_mace_mp_model_path, NffBatch)


class NffScaleMACE(ScaleShiftMACE):
    """Wrapper for the ScaleShiftMACE model that allows for direct
    forward passes in NFF
    """

    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        """NFF compatible ScaleShiftMACE

        Args:
            atomic_inter_scale (float): interaction energy scale
            atomic_inter_shift (float): interaction energy shift
        """
        super().__init__(atomic_inter_scale, atomic_inter_shift, **kwargs)

    def forward(
        self,
        batch: dict,
        training: bool = False,  # retained to keep the same args as the original MACE model forward
        compute_force: bool = True,
        requires_embedding=True,
        requires_stress=False,
        compute_virials: bool = False,
        compute_displacement: bool = False,
        **kwargs
    ) -> dict:  # noqa: W0221
        """Forward pass through the model

        Args:
            batch (dict): dictionary of Geom properties (energy, energy_grad, stress, etc.)
            kwargs: additional keyword arguments for the super().forward() function from MACE

        Returns:
            dict: dict of output from the forward pass
        """
        if isinstance(batch, dict):
            data = self.convert_batch_to_data(batch)
        else:
            data = batch
        output = super().forward(
            data,
            training=training,  # set the training mode to the value of the wrapper
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=requires_stress,
            compute_displacement=compute_displacement,
        )
        forces = output.pop("forces")
        node_features = output.pop("node_feats") # Node embedding
        if requires_embedding:
            output.update({"embedding": node_features})
        output.update({"energy_grad": -forces})
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
        if props["num_atoms"].dim() == 0:
            num_atoms = props["num_atoms"].unsqueeze(0)
        else:
            num_atoms = props["num_atoms"]
        cum_idx_list = [0] + torch.cumsum(num_atoms, 0).tolist()
        z_table = AtomicNumberTable([int(z) for z in self.atomic_numbers])

        dataset = []

        for i in range(num_atoms.shape[0]):
            node_idx = torch.arange(cum_idx_list[i], cum_idx_list[i + 1])
            positions = props.get("nxyz")[node_idx, 1:].detach().cpu().numpy()
            numbers = props.get("nxyz")[node_idx, 0].long().detach().cpu().numpy()

            if "cell" in props.keys():
                cell = props["cell"][3 * i : 3 * i + 3].detach().cpu().numpy()
            elif "lattice" in props.keys():
                cell = props["lattice"][3 * i : 3 * i + 3].detach().cpu().numpy()
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
            atomic_data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)
            atomic_data.elems = torch.tensor(numbers, dtype=torch.long)
            dataset.append(
                atomic_data
            )
        data = NffBatch.from_data_list(dataset).to(props["nxyz"].device)

        return data

    @property
    def num_params(self):
        """Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters())
    
    @property
    def num_trainable_params(self):
        """Count the number of parameters in the model.

        Returns:
            int: Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_init_kwargs(self) -> dict:
        """Get the initialization arguments from the model"""
        if isinstance(self.radial_embedding.bessel_fn, BesselBasis):
            radial_type = "bessel"
        elif isinstance(self.radial_embedding.bessel_fn, GaussianBasis):
            radial_type = "gaussian"
        
        init_kwargs = {
            "atomic_inter_scale": self.scale_shift.scale, 
            "atomic_inter_shift": self.scale_shift.shift,
            "r_max": self.r_max.item(),
            "num_bessel": self.radial_embedding.out_dim,
            "num_polynomial_cutoff": self.radial_embedding.cutoff_fn.p.item(),
            "max_ell": self.spherical_harmonics.irreps_out.lmax,
            "interaction_cls": type(self.interactions[1]),
            "interaction_cls_first": type(self.interactions[0]),
            "num_interactions": self.num_interactions.item(),
            "num_elements": self.node_embedding.linear.irreps_in.dim,
            "hidden_irreps": self.interactions[0].hidden_irreps,
            "MLP_irreps": self.readouts[-1].hidden_irreps,
            "atomic_energies": self.atomic_energies_fn.atomic_energies,
            "avg_num_neighbors": self.interactions[0].avg_num_neighbors,
            "atomic_numbers": self.atomic_numbers.tolist(),
            "correlation": self.products[0]
            .symmetric_contractions.contractions[0]
            .correlation,
            "gate": self.readouts[-1].non_linearity.acts[0].f,
            "radial_MLP": self.interactions[0].conv_tp_weights.hs[1:-1],
            "radial_type": radial_type
        }
        return init_kwargs

    def save(self, path: str):
        """Load a MACE model from a file or URL"""
        hparams = self.get_init_kwargs()
        state_dict = self.state_dict()
        torch.save({"init_params": hparams, "state_dict": state_dict}, path)

    @classmethod
    def load(cls, state_dict, **hparams):
        model = cls(**hparams)
        model.load_state_dict(state_dict=state_dict)
        return model

    @classmethod
    def from_file(cls, path: str) -> "NffScaleMACE":
        """Load the model from checkpoint created by pytorch lightning.

        Args:
            path (str): Path to the checkpoint file.

        Returns:
            InterAtomicPotential: The loaded model.
        """
        map_location = None if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=map_location)
        hparams = ckpt["init_params"]
        state_dict = ckpt["state_dict"]
        model = cls.load(state_dict, **hparams)
        return model

    @classmethod
    def load_foundations(cls, model: str = "medium", map_location: str = "cpu", default_dtype: str = ""):
        mace_model_path = get_mace_mp_model_path(model)
        mace_model = torch.load(mace_model_path, map_location=map_location)
        init_params = get_init_kwargs_from_model(mace_model)
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
                mace_model.double()
            elif default_dtype == "float32":
                mace_model.float()
        torch_tools.set_default_dtype(default_dtype)

        return cls.load(mace_model.state_dict(), **init_params)


def reduce_foundations(
    model_foundations: NffScaleMACE,
    table: Union[List, AtomicNumberTable],
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=1,
    num_conv_tp_weights=4,
    num_products=2,
    num_contraction=2
) -> "NffScaleMACE":
    """Reducing the model by extracting elements of interests
    Refer to the original paper to understand the architecture:
    "https://openreview.net/forum?id=YPpSngE-ZU"

    Args:
        model_foundations (NffScaleMACE): foundational model
        table (Union[List, AtomicNumberTable]): List of elements reduced from all preodic elements.
        load_readout (bool, optional): whether to restore the reduced model readouts. Defaults to False.
        use_shift (bool, optional): whether to restore the reduced model shift. Defaults to True.
        use_scale (bool, optional): whether to restore the reduced model scale. Defaults to True.
        max_L (int, optional): product blocks contraction max L. Defaults to 1.
        num_conv_tp_weights (int, optional): number of interactions. Defaults to 4.
        num_products (int, optional): number of products. Defaults to 2.
        num_contraction (int, optional): number of contraction. Defaults to 2.

    Returns:
        NffScaleMACE: reduced model
    """
    if isinstance(table, List):
        reduced_atomic_numbers = table
        table = get_atomic_number_table_from_zs(table)
    elif isinstance(AtomicNumberTable):
        reduced_atomic_numbers = [n for n in table.zs]
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = model_foundations.node_embedding.linear.weight.shape[0] // num_species_foundations
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]

    init_params = model_foundations.get_init_kwargs()
    atomic_energies = init_params["atomic_energies"]
    reduced_atomic_energies = atomic_energies[torch.tensor(indices_weights, dtype=torch.long)].clone()

    init_params.update({"atomic_energies": reduced_atomic_energies, "atomic_numbers": reduced_atomic_numbers, "num_elements": len(reduced_atomic_numbers)})
    model = NffScaleMACE(**init_params)
    assert model_foundations.r_max == model.r_max

    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )

    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors
        for j in range(num_conv_tp_weights):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(
                    model.interactions[i].conv_tp_weights, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].conv_tp_weights, layer_name
                    )
                    .weight[:num_radial, :]
                    .clone()
                )
            else:
                getattr(
                    model.interactions[i].conv_tp_weights, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model_foundations.interactions[i].conv_tp_weights, layer_name
                    ).weight.clone()
                )

        model.interactions[i].linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear.weight.clone()
        )
        if (
            model.interactions[i].__class__.__name__
            == "RealAgnosticResidualInteractionBlock"
        ):
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    num_species_foundations,
                    num_channels_foundation,
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )

    # Transferring products
    for i in range(num_products):  # Assuming 2 products modules
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            model.products[i].symmetric_contractions.contractions[
                j
            ].weights_max = torch.nn.Parameter(
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights_max[indices_weights, :, :]
                .clone()
            )

            for k in range(num_contraction):  # Assuming 2 weights in each contraction
                model.products[i].symmetric_contractions.contractions[j].weights[
                    k
                ] = torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights[k][indices_weights, :, :]
                    .clone()
                )

        model.products[i].linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

    if load_readout:
        # Transferring readouts
        model.readouts[0].linear.weight = torch.nn.Parameter(
            model_foundations.readouts[0].linear.weight.clone()
        )

        model.readouts[1].linear_1.weight = torch.nn.Parameter(
            model_foundations.readouts[1].linear_1.weight.clone()
        )

        model.readouts[1].linear_2.weight = torch.nn.Parameter(
            model_foundations.readouts[1].linear_2.weight.clone()
        )
    if model_foundations.scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.clone()
    return model


def restore_foundations(
    model: NffScaleMACE,
    model_foundations: NffScaleMACE,
    load_readout=True,
    use_shift=True,
    use_scale=True,
    max_L=1,
    num_conv_tp_weights=4,
    num_products=2,
    num_contraction=2
) -> "NffScaleMACE":
    """Restore back to foundational model from reduced model
    Refer to the original paper to understand the architecture:
    "https://openreview.net/forum?id=YPpSngE-ZU"

    Args:
        model (NffScaleMACE): reduced model
        model_foundations (NffScaleMACE): foundational model
        load_readout (bool, optional): whether to restore the reduced model readouts. Defaults to True.
        use_shift (bool, optional): whether to restore the reduced model shift. Defaults to True.
        use_scale (bool, optional): whether to restore the reduced model scale. Defaults to True.
        max_L (int, optional): product blocks contraction max L. Defaults to 2 for medium accuracy mp-mace.

    Returns:
        NffScaleMACE: All atom MACE model with updated model parameters
    """
    assert model_foundations.r_max == model.r_max
    assert model_foundations.radial_embedding.out_dim == model.radial_embedding.out_dim
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    new_z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    num_species_foundations = len(z_table.zs)
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_species = len(indices_weights)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    num_channels = (
        model.node_embedding.linear.weight.shape[0]
        // num_species
    )
    num_radial = model.radial_embedding.out_dim
    num_radial_foundations = model_foundations.radial_embedding.out_dim

    assert num_channels_foundation == num_channels
    assert num_radial_foundations == num_radial

    for i in range(int(model.num_interactions)):
        model_foundations.interactions[i].linear_up.weight = torch.nn.Parameter(
            model.interactions[i].linear_up.weight.clone()
        )
        model_foundations.interactions[i].avg_num_neighbors = model.interactions[
            i
        ].avg_num_neighbors
        for j in range(num_conv_tp_weights):  # Assuming 4 layers in conv_tp_weights,
            layer_name = f"layer{j}"
            if j == 0:
                getattr(
                    model_foundations.interactions[i].conv_tp_weights, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model.interactions[i].conv_tp_weights, layer_name
                    )
                    .weight[:num_radial, :]
                    .clone()
                )
            else:
                getattr(
                    model_foundations.interactions[i].conv_tp_weights, layer_name
                ).weight = torch.nn.Parameter(
                    getattr(
                        model.interactions[i].conv_tp_weights, layer_name
                    ).weight.clone()
                )

        model_foundations.interactions[i].linear.weight = torch.nn.Parameter(
            model.interactions[i].linear.weight.clone()
        )
        # Assuming 'model' and 'model_foundations' are instances of some torch.nn.Module
        # And assuming the other variables (num_channels_foundation, num_species_foundations, etc.) are correctly defined

        if (
            model.interactions[i].__class__.__name__
            == "RealAgnosticResidualInteractionBlock"
        ):
            for k, index in enumerate(indices_weights):
                # Get the original weights and apply transformation
                original_weights = model.interactions[i].skip_tp.weight.view(num_channels, num_species, num_channels)[:,k,:]
                transformed_weights = original_weights.flatten().clone() * (num_species_foundations / num_species) ** 0.5

                # Ensure the target tensor is appropriately resized
                # This step assumes you want to replace the weights directly. 
                # Adjust the view dimensions as necessary to match your actual model architecture.
                target_shape = (num_channels_foundation, num_species_foundations, num_channels_foundation)
                model_foundations.interactions[i].skip_tp.weight.data = model_foundations.interactions[i].skip_tp.weight.data.view(*target_shape)
                
                # Update the weights directly without wrapping them in torch.nn.Parameter
                model_foundations.interactions[i].skip_tp.weight.data[:,index,:] = transformed_weights.view_as(model_foundations.interactions[i].skip_tp.weight.data[:,index,:])
    # Transferring products
    for i in range(num_products):  # Assuming 2 products modules
        max_range = max_L + 1 if i == 0 else 1
        for j in range(max_range):  # Adjust `max_range` as per your specific case
            # Extract the entire weights tensor
            original_weights_max = model_foundations.products[i].symmetric_contractions.contractions[j].weights_max
            
            # Assuming 'indices_weights' is a list of indices you wish to update in 'original_weights'
            for k, index in enumerate(indices_weights):
                # Clone and prepare the new parameter from the source model
                new_weights_max = model.products[i].symmetric_contractions.contractions[j].weights_max[k, :, :].clone()
                
                # Replace the relevant slice of 'original_weights' with 'new_weight'
                # Note: This is done outside the parameter to avoid in-place modification errors
                original_weights_max.data[index, :, :] = torch.nn.Parameter(new_weights_max)

            original_weights_list = model_foundations.products[i].symmetric_contractions.contractions[j].weights
            for l in range(num_contraction): # Assuming 2 weights in each contractions
                original_weights = original_weights_list[l]
                for k, index in enumerate(indices_weights):
                    new_weights = model.products[i].symmetric_contractions.contractions[j].weights[l][k, :, :].clone()
                    original_weights.data[index, :, :] = torch.nn.Parameter(new_weights)

        model_foundations.products[i].linear.weight = torch.nn.Parameter(
            model.products[i].linear.weight.clone()
        )

    if load_readout:
        # Transferring readouts
        model_foundations.readouts[0].linear.weight = torch.nn.Parameter(
            model.readouts[0].linear.weight.clone()
        )

        model_foundations.readouts[1].linear_1.weight = torch.nn.Parameter(
            model.readouts[1].linear_1.weight.clone()
        )

        model_foundations.readouts[1].linear_2.weight = torch.nn.Parameter(
            model.readouts[1].linear_2.weight.clone()
        )
    if model.scale_shift is not None:
        if use_scale:
            model_foundations.scale_shift.scale = model.scale_shift.scale.clone()
        if use_shift:
            model_foundations.scale_shift.shift = model.scale_shift.shift.clone()
    return model_foundations