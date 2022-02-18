import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import switch_function
from typing import Optional

"""
computes electrostatic energy, switches between a constant value
and the true Coulomb law between cuton and cutoff
"""


class ElectrostaticEnergy(nn.Module):
    def __init__(
        self,
        ke: float = 14.399645351950548,
        cuton: float = 0.0,
        cutoff: float = 1.0,
        lr_cutoff: Optional[float] = None,
    ) -> None:
        super(ElectrostaticEnergy, self).__init__()
        self.ke = ke
        self.kehalf = ke / 2
        self.cuton = cuton
        self.cutoff = cutoff
        self.set_lr_cutoff(lr_cutoff)
        # should be turned on manually if the user knows what they are doing
        self.use_ewald_summation = False
        # set optional attributes to default value for jit compatibility
        self.alpha = 0.0
        self.alpha2 = 0.0
        self.two_pi = 2.0 * math.pi
        self.one_over_sqrtpi = 1 / math.sqrt(math.pi)
        self.register_buffer(
            "kmul", torch.Tensor(), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def set_lr_cutoff(self, lr_cutoff: Optional[float] = None) -> None:
        """ Change the long range cutoff. """
        self.lr_cutoff = lr_cutoff
        if self.lr_cutoff is not None:
            self.lr_cutoff2 = lr_cutoff ** 2
            self.two_div_cut = 2.0 / lr_cutoff
            self.rcutconstant = lr_cutoff / (lr_cutoff ** 2 + 1.0) ** (3.0 / 2.0)
            self.cutconstant = (2 * lr_cutoff ** 2 + 1.0) / (lr_cutoff ** 2 + 1.0) ** (
                3.0 / 2.0
            )
        else:
            self.lr_cutoff2 = None
            self.two_div_cut = None
            self.rcutconstant = None
            self.cutconstant = None

    def set_kmax(self, Nxmax: int, Nymax: int, Nzmax: int) -> None:
        """ Set integer reciprocal space cutoff for Ewald summation """
        kx = torch.arange(0, Nxmax + 1)
        kx = torch.cat([kx, -kx[1:]])
        ky = torch.arange(0, Nymax + 1)
        ky = torch.cat([ky, -ky[1:]])
        kz = torch.arange(0, Nzmax + 1)
        kz = torch.cat([kz, -kz[1:]])
        kmul = torch.cartesian_prod(kx, ky, kz)[1:]  # 0th entry is 0 0 0
        kmax = max(max(Nxmax, Nymax), Nzmax)
        self.register_buffer(
            "kmul", kmul[torch.sum(kmul ** 2, dim=-1) <= kmax ** 2], persistent=False
        )

    def set_alpha(self, alpha: Optional[float] = None) -> None:
        """ Set real space damping parameter for Ewald summation """
        if alpha is None:  # automatically determine alpha
            alpha = 4.0 / self.cutoff + 1e-3
        self.alpha = alpha
        self.alpha2 = alpha ** 2
        self.two_pi = 2.0 * math.pi
        self.one_over_sqrtpi = 1 / math.sqrt(math.pi)
        # print a warning if alpha is so small that the reciprocal space sum
        # might "leak" into the damped part of the real space coulomb interaction
        if alpha * self.cutoff < 4.0:  # erfc(4.0) ~ 1e-8
            print(
                "Warning: Damping parameter alpha is",
                alpha,
                "but probably should be at least",
                4.0 / self.cutoff,
            )

    def _real_space(
        self,
        N: int,
        q: torch.Tensor,
        rij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        if q.device.type == "cpu":  # indexing is faster on CPUs
            fac = self.kehalf * q[idx_i] * q[idx_j]
        else:  # gathering is faster on GPUs
            fac = self.kehalf * torch.gather(q, 0, idx_i) * torch.gather(q, 0, idx_j)
        f = switch_function(rij, self.cuton, self.cutoff)
        coulomb = 1.0 / rij
        damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
        pairwise = fac * (f * damped + (1 - f) * coulomb) * torch.erfc(self.alpha * rij)
        return q.new_zeros(N).index_add_(0, idx_i, pairwise)

    def _reciprocal_space(
        self,
        q: torch.Tensor,
        R: torch.Tensor,
        cell: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        # calculate k-space vectors
        box_length = torch.diagonal(cell, dim1=-2, dim2=-1)
        k = self.two_pi * self.kmul.unsqueeze(0) / box_length.unsqueeze(-2)
        # gaussian charge density
        k2 = torch.sum(k * k, dim=-1)  # squared length of k-vectors
        qg = torch.exp(-0.25 * k2 / self.alpha2) / k2
        # fourier charge density
        if k.device.type == "cpu":  # indexing is faster on CPUs
            dot = torch.sum(k[batch_seg] * R.unsqueeze(-2), dim=-1)
        else:  # gathering is faster on GPUs
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])
            dot = torch.sum(torch.gather(k, 0, b) * R.unsqueeze(-2), dim=-1)
        q_real = q.new_zeros(num_batch, dot.shape[-1]).index_add_(
            0, batch_seg, q.unsqueeze(-1) * torch.cos(dot)
        )
        q_imag = q.new_zeros(num_batch, dot.shape[-1]).index_add_(
            0, batch_seg, q.unsqueeze(-1) * torch.sin(dot)
        )
        qf = q_real ** 2 + q_imag ** 2
        # reciprocal energy
        e_reciprocal = (
            self.two_pi / torch.prod(box_length, dim=1) * torch.sum(qf * qg, dim=-1)
        )
        # self interaction correction
        q2 = q * q
        e_self = self.alpha * self.one_over_sqrtpi * q2
        # spread reciprocal energy over atoms (to get an atomic contributions)
        w = q2 + eps  # epsilon is added to prevent division by zero
        wnorm = w.new_zeros(num_batch).index_add_(0, batch_seg, w)
        if w.device.type == "cpu":  # indexing is faster on CPUs
            w = w / wnorm[batch_seg]
            e_reciprocal = w * e_reciprocal[batch_seg]
        else:  # gathering is faster on GPUs
            w = w / torch.gather(wnorm, 0, batch_seg)
            e_reciprocal = w * torch.gather(e_reciprocal, 0, batch_seg)
        return self.ke * (e_reciprocal - e_self)

    def _ewald(
        self,
        N: int,
        q: torch.Tensor,
        R: torch.Tensor,
        rij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
    ) -> torch.Tensor:
        e_real = self._real_space(N, q, rij, idx_i, idx_j)
        e_reciprocal = self._reciprocal_space(q, R, cell, num_batch, batch_seg)
        return e_real + e_reciprocal

    def _coulomb(
        self,
        N: int,
        q: torch.Tensor,
        rij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        if q.device.type == "cpu":  # indexing is faster on CPUs
            fac = self.kehalf * q[idx_i] * q[idx_j]
        else:  # gathering is faster on GPUs
            fac = self.kehalf * torch.gather(q, 0, idx_i) * torch.gather(q, 0, idx_j)
        f = switch_function(rij, self.cuton, self.cutoff)
        if self.lr_cutoff is None:
            coulomb = 1.0 / rij
            damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
        else:
            coulomb = torch.where(
                rij < self.lr_cutoff,
                1.0 / rij + rij / self.lr_cutoff2 - self.two_div_cut,
                torch.zeros_like(rij),
            )
            damped = torch.where(
                rij < self.lr_cutoff,
                1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
                + rij * self.rcutconstant
                - self.cutconstant,
                torch.zeros_like(rij),
            )
        pairwise = fac * (f * damped + (1 - f) * coulomb)
        return q.new_zeros(N).index_add_(0, idx_i, pairwise)

    def forward(
        self,
        N: int,
        q: torch.Tensor,
        rij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        R: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_ewald_summation:
            assert R is not None
            assert cell is not None
            assert batch_seg is not None
            return self._ewald(N, q, R, rij, idx_i, idx_j, cell, num_batch, batch_seg)
        else:
            return self._coulomb(N, q, rij, idx_i, idx_j)
