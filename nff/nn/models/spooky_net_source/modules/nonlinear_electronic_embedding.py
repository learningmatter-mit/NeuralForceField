import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention
from .residual_mlp import ResidualMLP
from .shifted_softplus import ShiftedSoftplus
from .swish import Swish
from typing import Optional


class NonlinearElectronicEmbedding(nn.Module):
    """
    Block for updating atomic features through nonlocal interactions with the
    electrons.

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
        self, num_features: int, num_residual: int, activation: str = "swish"
    ) -> None:
        """ Initializes the NonlinearElectronicEmbedding class. """
        super(NonlinearElectronicEmbedding, self).__init__()
        self.linear_q = nn.Linear(num_features, num_features, bias=False)
        self.featurize_k = nn.Linear(1, num_features)
        self.resblock_k = ResidualMLP(
            num_features, num_residual, activation=activation, zero_init=True
        )
        self.featurize_v = nn.Linear(1, num_features, bias=False)
        self.resblock_v = ResidualMLP(
            num_features,
            num_residual,
            activation=activation,
            zero_init=True,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        nn.init.orthogonal_(self.linear_q.weight)
        nn.init.orthogonal_(self.featurize_k.weight)
        nn.init.zeros_(self.featurize_k.bias)
        nn.init.orthogonal_(self.featurize_v.weight)

    def forward(
        self,
        x: torch.Tensor,
        E: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        e = E.unsqueeze(-1)
        q = self.linear_q(x)  # queries
        k = self.resblock_k(self.featurize_k(e))[batch_seg]  # keys
        v = self.resblock_v(self.featurize_v(e))[batch_seg]  # values
        # dot product
        dot = torch.sum(k * q, dim=-1)
        # determine maximum dot product (for numerics)
        if num_batch > 1:
            if mask is None:
                mask = (
                    nn.functional.one_hot(batch_seg)
                    .to(dtype=x.dtype, device=x.device)
                    .transpose(-1, -2)
                )
            tmp = dot.view(1, -1).expand(num_batch, -1)
            tmp, _ = torch.max(mask * tmp, dim=-1)
            if tmp.device.type == "cpu":  # indexing is faster on CPUs
                maximum = tmp[batch_seg]
            else:  # gathering is faster on GPUs
                maximum = torch.gather(tmp, 0, batch_seg)
        else:
            maximum = torch.max(dot)
        # attention
        d = k.shape[-1]
        a = torch.exp((dot - maximum) / d ** 0.5)

        anorm = a.new_zeros(num_batch).index_add_(0, batch_seg, a)
        if a.device.type == "cpu":  # indexing is faster on CPUs
            anorm = anorm[batch_seg]
        else:  # gathering is faster on GPUs
            anorm = torch.gather(anorm, 0, batch_seg)
        return (a / (anorm + eps)).unsqueeze(-1) * v
