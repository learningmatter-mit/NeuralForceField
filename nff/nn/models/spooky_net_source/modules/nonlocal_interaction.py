import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention
from .residual_mlp import ResidualMLP
from typing import Optional


class NonlocalInteraction(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with all
    atoms.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre_i (int):
            Number of residual blocks applied to atomic features in i branch
            (central atoms) before computing the interaction.
        num_residual_pre_j (int):
            Number of residual blocks applied to atomic features in j branch
            (neighbouring atoms) before computing the interaction.
        num_residual_post (int):
            Number of residual blocks applied to interaction features.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        num_features: int,
        num_residual_q: int,
        num_residual_k: int,
        num_residual_v: int,
        activation: str = "swish",
    ) -> None:
        """ Initializes the NonlocalInteraction class. """
        super(NonlocalInteraction, self).__init__()
        self.resblock_q = ResidualMLP(
            num_features, num_residual_q, activation=activation, zero_init=True
        )
        self.resblock_k = ResidualMLP(
            num_features, num_residual_k, activation=activation, zero_init=True
        )
        self.resblock_v = ResidualMLP(
            num_features, num_residual_v, activation=activation, zero_init=True
        )
        self.attention = Attention(num_features, num_features, num_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def forward(
        self,
        x: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        q = self.resblock_q(x)  # queries
        k = self.resblock_k(x)  # keys
        v = self.resblock_v(x)  # values
        return self.attention(q, k, v, num_batch, batch_seg, mask)
