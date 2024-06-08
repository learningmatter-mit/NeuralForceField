"""
The transfer learning module provides functions to fine tune
a pretrained model with a new given dataloader. It relies
on pretrained models, which can be loaded from checkpoints
or best models using the Trainer class.

Last refactored 2024-03-07 by Alex Hoffman
"""

from typing import List

import torch


class LayerFreezer:
    """General class to handle freezing layers in models"""

    def freeze_parameters(self, model: torch.nn.Module) -> None:
        """
        Freezes all parameters from a given model.

        Args:
            model (any of nff.nn.models)
        """
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self, module: torch.nn.Module) -> None:
        """Unfreeze parameters in the module

        Args:
            module (torch.nn.Module): module to unfreeze
        """
        for param in module.parameters():
            param.requires_grad = True

    def custom_unfreeze(self, model: torch.nn.Module, custom_layers: List[str]) -> None:
        """Unfreeze parameters from a custom list.

        Args:
            model (torch.nn.Module): model to be transfer learned
            custom_unfreeze (List[str]): list of layers to unfreeze specified by
                the user. The items in the custom list should be strings
                from the names of the parameters in the model, which can be obtained
                from list(model.named_parameters())
        """
        for module in model.named_parameters():
            if module[0] in custom_layers:
                module[1].requires_grad = True

    def unfreeze_readout(self, model: torch.nn.Module) -> None:
        """
        Unfreezes the parameters from the readout layers.

        Args:
            model (any of nff.nn.models): the model to be transfer learned
        """
        self.unfreeze_parameters(model.atomwisereadout)

    def model_tl(
        self,
        model: torch.nn.Module,
        freeze_gap_embedding: bool,
        freeze_pooling: bool,
        freeze_skip: bool,
        custom_layers: List[str],
        **kwargs,
    ) -> None:
        """
        Function to transfer learn a model. Defined in the subclasses.
        """
        pass


class PainnLayerFreezer(LayerFreezer):
    """Class to handle freezing in PaiNN models"""

    def unfreeze_painn_readout(self, model: torch.nn.Module, freeze_skip: bool) -> None:
        """Unfreeze the readout layers in a PaiNN model.

        Args:
            model (torch.nn.Module): the model to be transfer learned
            freeze_skip (bool): if true, keep all but the last readout layer frozen.
        """
        num_readouts = len(model.readout_blocks)
        unfreeze_skip = not freeze_skip

        for i, block in enumerate(model.readout_blocks):
            if unfreeze_skip:
                self.unfreeze_parameters(block)
            elif i == num_readouts - 1:
                self.unfreeze_parameters(block)

    def unfreeze_painn_pooling(self, model: torch.nn.Module) -> None:
        """Unfreeze the pooling layers in a PaiNN model.

        Args:
            model (torch.nn.Module): model to be transfer learned
        """
        for module in model.pool_dic.values():
            self.unfreeze_parameters(module)

    def model_tl(
        self,
        model: torch.nn.Module,
        freeze_gap_embedding: bool,  # unused for regular PaiNN
        freeze_pooling: bool,
        freeze_skip: bool,
        custom_layers: List[str],  # unused for PaiNN
        **kwargs,
    ) -> None:
        """Function to transfer learn a PaiNN model.

        Args:
            model (torch.nn.Module): model to be transfer learned
            freeze_gap_embedding (bool): unused in this class, only
                for diabatic models
            freeze_pooling (bool): if true, keep all pooling layers frozen
            freeze_skip (bool): if true, keep all but the last readout layer frozen
            custom_layers (List[str]): list of layers to unfreeze specified by the user
                that is different from the default. Unused in this class.
        """
        self.freeze_parameters(model)
        self.unfreeze_painn_readout(model=model, freeze_skip=freeze_skip)
        unfreeze_pool = not freeze_pooling
        if unfreeze_pool:
            self.unfreeze_painn_pooling(model)


class PainnDiabatLayerFreezer(PainnLayerFreezer):
    """Class to handle freezing layers in PaiNN models with diabatic readout."""

    def unfreeze_diabat_readout(
        self, model: torch.nn.Module, freeze_gap_embedding: bool
    ) -> None:
        """Unfreeze the diabatic readout layers in a PaiNN model.

        Args:
            model (torch.nn.Module): model to be transfer learned
            freeze_gap_embedding (bool): if true, keep the gap embedding frozen
        """
        cross_talk = model.diabatic_readout.cross_talk
        unfreeze_gap = not freeze_gap_embedding
        if not cross_talk:
            return
        for module in cross_talk.coupling_modules:
            if hasattr(module, "readout"):
                self.unfreeze_parameters(module.readout)
            if hasattr(module, "featurizer") and unfreeze_gap:
                self.unfreeze_parameters(module.featurizer)

    def model_tl(
        self,
        model: torch.nn.Module,
        freeze_gap_embedding: bool,
        freeze_pooling: bool,
        freeze_skip: bool,
        custom_layers: List[str],  # unused for PaiNN
        **kwargs,
    ):
        """Function to transfer learn a PaiNN model with diabatic readout.

        Args:
            model (torch.nn.Module): model to be transfer learned
            freeze_gap_embedding (bool): if true, keep the gap embedding frozen
            freeze_pooling (bool): if true, keep all pooling layers frozen
            freeze_skip (bool): if true, keep all but the last readout layer frozen
            custom_layers (List[str]): list of layers to unfreeze specified by the user
                that is different from the default. Unused in this class.
        """
        self.freeze_parameters(model)
        self.unfreeze_painn_readout(model=model, freeze_skip=freeze_skip)
        self.unfreeze_diabat_readout(
            model=model, freeze_gap_embedding=freeze_gap_embedding
        )

        unfreeze_pool = not freeze_pooling
        if unfreeze_pool:
            self.unfreeze_painn_pooling(model)


class MaceLayerFreezer(LayerFreezer):
    """Class to handle freezing layers in MACE models."""

    def unfreeze_mace_interaction_linears(self, model: torch.nn.Module) -> None:
        """Unfreeze the linear readout layer from the interaction blocks in
        a MACE model.

        Args:
            model (torch.nn.Module): model to be transfer learned
        """
        interaction_linears = [
            f"interactions.{i}.linear.weight"
            for i in range(model.num_interactions.item())
        ]
        self.custom_unfreeze(model, interaction_linears)

    def unfreeze_mace_produce_linears(self, model: torch.nn.Module) -> None:
        """Unfreeze the linear readout layer from the interaction blocks in
        a MACE model.

        Args:
            model (torch.nn.Module): model to be transfer learned
        """
        product_linears = [
            f"products.{i}.linear.weight"
            for i in range(model.num_interactions.item())
        ]
        self.custom_unfreeze(model, product_linears)

    def unfreeze_mace_pooling(self, model: torch.nn.Module) -> None:
        """Unfreeze the pooling layers in a MACE model (called the products layer)

        Args:
            model (torch.nn.Module): model to be transfer learned
        """
        for module in model.products:
            self.unfreeze_parameters(module)

    def unfreeze_mace_readout(self, model: torch.nn.Module, freeze_skip: bool = False):
        """Unfreeze the readout layers in a MACE model.

        Args:
            model (): _description_
            freeze_skip (bool, optional): If true, keep all but the last readout layer
                frozen. Defaults to False.
        """
        num_readouts = len(model.readouts)
        unfreeze_skip = not freeze_skip

        for i, block in enumerate(model.readouts):
            if unfreeze_skip:
                self.unfreeze_parameters(block)
            elif i == num_readouts - 1:
                self.unfreeze_parameters(block)

    def model_tl(
        self,
        model: torch.nn.Module,
        freeze_gap_embedding: bool = False,  # unused for MACE
        freeze_interactions: bool = True,
        freeze_products: bool = False,
        freeze_pooling: bool = True,
        freeze_skip: bool = False,
        custom_layers: List[str] = [],
        **kwargs,
    ) -> None:
        """Function to transfer learn a MACE model.

        Args:
            model (torch.nn.Module): MACE model
            freeze_gap_embedding (bool, optional): Unused for MACE, inherited from
                parent class for consistency with the diabatic models.
            freeze_pooling (bool, optional): If true, keep all pooling layers frozen.
                Defaults to True.
            freeze_skip (bool, optional): If true, keep all but the last readout layer
                frozen. Defaults to False.
            custom_layers (List[str]): list of layers to unfreeze specified by the user
                that is different from the default. From the output of
                list(model.named_parameters())[:]
        """
        self.freeze_parameters(model)
        if custom_layers:
            self.custom_unfreeze(model, custom_layers)
        else:
            self.unfreeze_mace_readout(model, freeze_skip=freeze_skip)
            unfreeze_pool = not freeze_pooling
            if unfreeze_pool:
                self.unfreeze_mace_pooling(model)
            if not freeze_interactions:
                self.unfreeze_mace_interaction_linears(model)
            if not freeze_products:
                self.unfreeze_mace_produce_linears(model)


class ChgnetLayerFreezer(LayerFreezer):
    """Class to handle freezing layers in Chgnet models.

    CHGNet operates slightly differently than other models. The default layers that
    this class freezes are adapted from this tutorial in the CHGNet repository:
    https://github.com/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb
    (accessed 2024-03-09)
    """

    def unfreeze_chgnet_last_atom_conv_layer(self, model: torch.nn.Module) -> None:
        """Unfreeze the pooling layers in a CHGNet model.

        Args:
            model (torch.nn.Module): model to be transfer learned
        """
        module = model.atom_conv_layers[-1]
        self.unfreeze_parameters(module)

    def unfreeze_chgnet_pooling(self, model: torch.nn.Module) -> None:
        """Unfreeze the "pooling" layers after the representation layers
        in a CHGNet model.

        Args:
            model (torch.nn.Module): model to be transfer learned
        """
        self.unfreeze_parameters(model.pooling)

    def unfreeze_chgnet_readout(
        self, model: torch.nn.Module, freeze_skip: bool = False
    ) -> None:
        """Unfreeze the "site_wise", "readout_norm", and last MLP layers
        in a CHGNet model. Similar to readout layers in other models.

        Args:
            model (torch.nn.Module): model to be transfer learned
            freeze_skip (bool, optional): If true, keep all but the last layer
                frozen. Defaults to False.
        """
        for module in [model.site_wise, model.readout_norm]:
            self.unfreeze_parameters(module)

        num_readouts = len(model.mlp.layers)
        unfreeze_skip = not freeze_skip

        for i, block in enumerate(model.mlp.layers):
            if unfreeze_skip:
                self.unfreeze_parameters(block)
            elif i == num_readouts - 1:
                self.unfreeze_parameters(block)

    def model_tl(
        self,
        model: torch.nn.Module,
        freeze_gap_embedding: bool = False,  # unused for CHGNet
        freeze_pooling: bool = False,  # suggested default from CHGNet repo
        freeze_skip: bool = False,
        custom_layers: List[str] = [],
        **kwargs,
    ) -> None:
        """Function to transfer learn a CHGNet model. Freezes all but
        the last readout layer and the pooling layers by default.

        Args:
            model (torch.nn.Module): model to be transfer learned
            freeze_gap_embedding (bool): unused for CHGNet but inherited from parent class
                for consistency with diabatic model
            freeze_pooling (bool): if true, keep all pooling layers frozen
            freeze_skip (bool): if true, keep all but the last readout layer frozen
            custom_layers (List[str]): list of layers to unfreeze specified by the user
                that is different from the default. From the output of
                list(model.named_parameters())[:]
        """
        self.freeze_parameters(model)
        if custom_layers:
            self.custom_unfreeze(model, custom_layers)
        else:
            self.unfreeze_chgnet_readout(model, freeze_skip=freeze_skip)
            unfreeze_pool = not freeze_pooling
            if unfreeze_pool:
                self.unfreeze_chgnet_last_atom_conv_layer(model)
                self.unfreeze_chgnet_pooling(model)
