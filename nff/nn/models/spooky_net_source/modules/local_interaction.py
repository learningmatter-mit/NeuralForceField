import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_mlp import ResidualMLP


class LocalInteraction(nn.Module):
    """
    Block for updating atomic features through local interactions with
    neighboring atoms (message-passing).

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
        num_basis_functions: int,
        num_residual_x: int,
        num_residual_s: int,
        num_residual_p: int,
        num_residual_d: int,
        num_residual: int,
        activation: str = "swish",
    ) -> None:
        """ Initializes the LocalInteraction class. """
        super(LocalInteraction, self).__init__()
        self.radial_s = nn.Linear(num_basis_functions, num_features, bias=False)
        self.radial_p = nn.Linear(num_basis_functions, num_features, bias=False)
        self.radial_d = nn.Linear(num_basis_functions, num_features, bias=False)
        self.resblock_x = ResidualMLP(num_features, num_residual_x, activation)
        self.resblock_s = ResidualMLP(num_features, num_residual_s, activation)
        self.resblock_p = ResidualMLP(num_features, num_residual_p, activation)
        self.resblock_d = ResidualMLP(num_features, num_residual_d, activation)
        self.projection_p = nn.Linear(num_features, 2 * num_features, bias=False)
        self.projection_d = nn.Linear(num_features, 2 * num_features, bias=False)
        self.resblock = ResidualMLP(
            num_features, num_residual, activation, zero_init=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters. """
        nn.init.orthogonal_(self.radial_s.weight)
        nn.init.orthogonal_(self.radial_p.weight)
        nn.init.orthogonal_(self.radial_d.weight)
        nn.init.orthogonal_(self.projection_p.weight)
        nn.init.orthogonal_(self.projection_d.weight)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        pij: torch.Tensor,
        dij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.
        P: Number of atom pairs.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        rbf (FloatTensor [N, num_basis_functions]):
            Values of the radial basis functions for the pairwise distances.
        idx_i (LongTensor [P]):
            Index of atom i for all atomic pairs ij. Each pair must be
            specified as both ij and ji.
        idx_j (LongTensor [P]):
            Same as idx_i, but for atom j.
        """
        # interaction functions
        gs = self.radial_s(rbf)
        gp = self.radial_p(rbf).unsqueeze(-2) * pij.unsqueeze(-1)
        gd = self.radial_d(rbf).unsqueeze(-2) * dij.unsqueeze(-1)
        # atom featurizations
        xx = self.resblock_x(x)
        xs = self.resblock_s(x)
        xp = self.resblock_p(x)
        xd = self.resblock_d(x)
        # collect neighbors
        if x.device.type == "cpu":  # indexing is faster on CPUs
            xs = xs[idx_j]  # L=0
            xp = xp[idx_j]  # L=1
            xd = xd[idx_j]  # L=2
        else:  # gathering is faster on GPUs
            j = idx_j.view(-1, 1).expand(-1, x.shape[-1])  # index for gathering
            xs = torch.gather(xs, 0, j)  # L=0
            xp = torch.gather(xp, 0, j)  # L=1
            xd = torch.gather(xd, 0, j)  # L=2
        # sum over neighbors
        pp = x.new_zeros(x.shape[0], pij.shape[-1], x.shape[-1])
        dd = x.new_zeros(x.shape[0], dij.shape[-1], x.shape[-1])
        s = xx.index_add(0, idx_i, gs * xs)  # L=0
        p = pp.index_add_(0, idx_i, gp * xp.unsqueeze(-2))  # L=1
        d = dd.index_add_(0, idx_i, gd * xd.unsqueeze(-2))  # L=2
        # project tensorial features to scalars
        pa, pb = torch.split(self.projection_p(p), p.shape[-1], dim=-1)
        da, db = torch.split(self.projection_d(d), d.shape[-1], dim=-1)
        return self.resblock(s + (pa * pb).sum(-2) + (da * db).sum(-2))
