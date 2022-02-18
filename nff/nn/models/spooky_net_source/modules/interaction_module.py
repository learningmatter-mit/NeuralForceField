import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_stack import ResidualStack
from .local_interaction import LocalInteraction
from .nonlocal_interaction import NonlocalInteraction
from .residual_mlp import ResidualMLP
from typing import Tuple, Optional


class InteractionModule(nn.Module):
    """
    InteractionModule of SpookyNet, which computes a single iteration.

    Arguments:
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_residual_pre (int):
            Number of residual blocks applied to atomic features before
            interaction with neighbouring atoms.
        num_residual_post (int):
            Number of residual blocks applied to atomic features after
            interaction with neighbouring atoms.
        num_residual_pre_local_i (int):
            Number of residual blocks applied to atomic features in i branch
            (central atoms) before computing the local interaction.
        num_residual_pre_local_j (int):
            Number of residual blocks applied to atomic features in j branch
            (neighbouring atoms) before computing the local interaction.
        num_residual_post_local (int):
            Number of residual blocks applied to interaction features.
        num_residual_output (int):
            Number of residual blocks applied to atomic features in output
            branch.
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
    """

    def __init__(
        self,
        num_features: int,
        num_basis_functions: int,
        num_residual_pre: int,
        num_residual_local_x: int,
        num_residual_local_s: int,
        num_residual_local_p: int,
        num_residual_local_d: int,
        num_residual_local: int,
        num_residual_nonlocal_q: int,
        num_residual_nonlocal_k: int,
        num_residual_nonlocal_v: int,
        num_residual_post: int,
        num_residual_output: int,
        activation: str = "swish",
    ) -> None:
        """ Initializes the InteractionModule class. """
        super(InteractionModule, self).__init__()
        # initialize modules
        self.local_interaction = LocalInteraction(
            num_features=num_features,
            num_basis_functions=num_basis_functions,
            num_residual_x=num_residual_local_x,
            num_residual_s=num_residual_local_s,
            num_residual_p=num_residual_local_p,
            num_residual_d=num_residual_local_d,
            num_residual=num_residual_local,
            activation=activation,
        )
        self.nonlocal_interaction = NonlocalInteraction(
            num_features=num_features,
            num_residual_q=num_residual_nonlocal_q,
            num_residual_k=num_residual_nonlocal_k,
            num_residual_v=num_residual_nonlocal_v,
            activation=activation,
        )
        self.residual_pre = ResidualStack(num_features, num_residual_pre, activation)
        self.residual_post = ResidualStack(num_features, num_residual_post, activation)
        self.resblock = ResidualMLP(
            num_features, num_residual_output, activation=activation
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        pij: torch.Tensor,
        dij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate all modules in the block.
        N: Number of atoms.
        P: Number of atom pairs.
        B: Batch size (number of different molecules).

        Arguments:
            x (FloatTensor [N, num_features]):
                Latent atomic feature vectors.
            rbf (FloatTensor [P, num_basis_functions]):
                Values of the radial basis functions for the pairwise distances.
            idx_i (LongTensor [P]):
                Index of atom i for all atomic pairs ij. Each pair must be
                specified as both ij and ji.
            idx_j (LongTensor [P]):
                Same as idx_i, but for atom j.
            num_batch (int):
                Batch size (number of different molecules).
            batch_seg (LongTensor [N]):
                Index for each atom that specifies to which molecule in the
                batch it belongs.
        Returns:
            x (FloatTensor [N, num_features]):
                Updated latent atomic feature vectors.
            y (FloatTensor [N, num_features]):
                Contribution to output atomic features (environment
                descriptors).
        """
        x = self.residual_pre(x)
        l = self.local_interaction(x, rbf, pij, dij, idx_i, idx_j)
        n = self.nonlocal_interaction(x, num_batch, batch_seg, mask)
        x = self.residual_post(x + l + n)
        return x, self.resblock(x)
