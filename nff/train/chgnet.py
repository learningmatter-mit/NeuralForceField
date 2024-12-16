from typing import Dict, Iterable, Union

import torch
from chgnet.trainer.trainer import CombinedLoss
from torch import Tensor

from nff.utils.cuda import detach


class CombinedLossNFF(CombinedLoss):
    """Wrapper for the combined loss function that maps keys from NFF to CHGNet keys."""

    def __init__(self, *args, key_mappings=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not key_mappings:
            # map from NFF keys to CHGNet keys
            self.key_mappings = {
                "energy": "e",
                "energy_grad": "f",
                "stress": "s",
                "magmom": "m",
                "num_atoms": "atoms_per_graph",
            }
            self.negate_keys = ("energy_grad",)
            self.split_keys = ("energy_grad", "stress", "magmom")

    def forward(self, targets: Dict[str, Tensor], predictions: Dict[str, Tensor], key_style: str = "nff"):
        """
        Forward pass for the combined loss function.
        Args:
            targets (dict): the targets of the model
            predictions (dict): the outputs of the model
        Returns:
            torch.Tensor: the combined loss
        """
        # TODO feels super hacky and unnecessary since we're breaking down the values and negating them again
        if key_style not in ["nff", "chgnet"]:
            raise ValueError("key_style must be either 'nff' or 'chgnet'")

        targets = {k: self.split_props(k, v, detach(targets["num_atoms"]).tolist()) for k, v in targets.items()}
        predictions = {
            k: self.split_props(
                k,
                v,
                detach(
                    predictions["num_atoms"]).tolist()) for k,
            v in predictions.items()}

        if key_style == "nff":
            targets = {self.key_mappings.get(k, k): self.negate_value(k, v) for k, v in targets.items()}
            predictions = {self.key_mappings.get(k, k): self.negate_value(k, v) for k, v in predictions.items()}

        out = super().forward(targets, predictions)
        loss = out["loss"]

        return loss

    def negate_value(self, key: str, value: Iterable) -> Union[list, Tensor]:
        """Negate the value if the key is in the negate_keys list.

        Args:
            key (str): the key
            value (Iterable): the value

        Returns:
            Union[list, Tensor]: the negated value
        """
        if key in self.negate_keys:
            if not isinstance(value, Tensor):
                return [-x for x in value]
            return -value
        return value

    def split_props(
        self, key: str, value: Union[list, Tensor], num_atoms: Union[list, Tensor]
    ) -> Union[list, Tensor]:
        """Split the properties if the key is in the split_keys list.

        Args:
            key (str): the key
            value (Union[list, Tensor]): the value
            num_atoms (Union[list, Tensor]): the number of atoms in the system

        Returns:
            Union[list, Tensor]: the split value
        """
        if key in self.split_keys:
            return torch.split(value, num_atoms)
        return value
