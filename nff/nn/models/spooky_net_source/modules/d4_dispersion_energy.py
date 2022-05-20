import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import softplus_inverse, switch_function
from typing import Tuple, Optional

"""
computes D4 dispersion energy
HF      s6=1.00000000, s8=1.61679827, a1=0.44959224, a2=3.35743605
"""


class D4DispersionEnergy(nn.Module):
    def __init__(
        self,
        cutoff: Optional[float] = None,
        s6: float = 1.00000000,
        s8: float = 1.61679827,
        a1: float = 0.44959224,
        a2: float = 3.35743605,
        g_a: float = 3.0,
        g_c: float = 2.0,
        k2: float = 1.3333333333333333,  # 4/3
        k4: float = 4.10451,
        k5: float = 19.08857,
        k6: float = 254.5553148552,  # 2*11.28174**2
        kn: float = 7.5,
        wf: float = 6.0,
        Zmax: int = 87,
        Bohr: float = 0.5291772105638411,  # conversion to Bohr
        Hartree: float = 27.211386024367243,  # conversion to Hartree
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """ Initializes the D4DispersionEnergy class. """
        super(D4DispersionEnergy, self).__init__()
        # Grimme's D4 dispersion is only parametrized up to Rn (Z=86)
        assert Zmax <= 87
        # trainable parameters
        self.register_parameter(
            "_s6", nn.Parameter(softplus_inverse(s6), requires_grad=False)
        )  # s6 is usually not fitted (correct long-range)
        self.register_parameter(
            "_s8", nn.Parameter(softplus_inverse(s8), requires_grad=True)
        )
        self.register_parameter(
            "_a1", nn.Parameter(softplus_inverse(a1), requires_grad=True)
        )
        self.register_parameter(
            "_a2", nn.Parameter(softplus_inverse(a2), requires_grad=True)
        )
        self.register_parameter(
            "_scaleq", nn.Parameter(softplus_inverse(1.0), requires_grad=True)
        )  # for scaling charges of reference systems
        # D4 constants
        self.Zmax = Zmax
        self.convert2Bohr = 1 / Bohr
        self.convert2eV = 0.5 * Hartree  # factor of 0.5 prevents double counting
        self.convert2Angstrom3 = Bohr ** 3
        self.convert2eVAngstrom6 = Hartree * Bohr ** 6
        self.set_cutoff(cutoff)
        self.g_a = g_a
        self.g_c = g_c
        self.k2 = k2
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kn = kn
        self.wf = wf
        # load D4 data
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d4data")
        self.register_buffer(
            "refsys",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "refsys.pth"))[:Zmax],
        )
        self.register_buffer(
            "zeff", torch.load(os.path.join(directory, "zeff.pth"))[:Zmax]  # [Zmax]
        )
        self.register_buffer(
            "refh",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "refh.pth"))[:Zmax],
        )
        self.register_buffer(
            "sscale", torch.load(os.path.join(directory, "sscale.pth"))  # [18]
        )
        self.register_buffer(
            "secaiw", torch.load(os.path.join(directory, "secaiw.pth"))  # [18,23]
        )
        self.register_buffer(
            "gam", torch.load(os.path.join(directory, "gam.pth"))[:Zmax]  # [Zmax]
        )
        self.register_buffer(
            "ascale",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "ascale.pth"))[:Zmax],
        )
        self.register_buffer(
            "alphaiw",  # [Zmax,max_nref,23]
            torch.load(os.path.join(directory, "alphaiw.pth"))[:Zmax],
        )
        self.register_buffer(
            "hcount",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "hcount.pth"))[:Zmax],
        )
        self.register_buffer(
            "casimir_polder_weights",  # [23]
            torch.load(os.path.join(directory, "casimir_polder_weights.pth"))[:Zmax],
        )
        self.register_buffer(
            "rcov", torch.load(os.path.join(directory, "rcov.pth"))[:Zmax]  # [Zmax]
        )
        self.register_buffer(
            "en", torch.load(os.path.join(directory, "en.pth"))[:Zmax]  # [Zmax]
        )
        self.register_buffer(
            "ncount_mask",  # [Zmax,max_nref,max_ncount]
            torch.load(os.path.join(directory, "ncount_mask.pth"))[:Zmax],
        )
        self.register_buffer(
            "ncount_weight",  # [Zmax,max_nref,max_ncount]
            torch.load(os.path.join(directory, "ncount_weight.pth"))[:Zmax],
        )
        self.register_buffer(
            "cn",  # [Zmax,max_nref,max_ncount]
            torch.load(os.path.join(directory, "cn.pth"))[:Zmax],
        )
        self.register_buffer(
            "fixgweights",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "fixgweights.pth"))[:Zmax],
        )
        self.register_buffer(
            "refq",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "refq.pth"))[:Zmax],
        )
        self.register_buffer(
            "sqrt_r4r2",  # [Zmax]
            torch.load(os.path.join(directory, "sqrt_r4r2.pth"))[:Zmax],
        )
        self.register_buffer(
            "alpha",  # [Zmax,max_nref,23]
            torch.load(os.path.join(directory, "alpha.pth"))[:Zmax],
        )
        self.max_nref = self.refsys.size(-1)
        self._compute_refc6()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def set_cutoff(self, cutoff: Optional[float] = None) -> None:
        """ Can be used to change the cutoff. """
        if cutoff is None:
            self.cutoff = None
            self.cuton = None
        else:
            self.cutoff = cutoff * self.convert2Bohr
            self.cuton = self.cutoff - 1.0 / self.convert2Bohr

    def _compute_refc6(self) -> None:
        """
        Function to compute the refc6 tensor. Important: If the charges of
        reference systems are scaled and the scaleq parameter changes (e.g.
        during training), then the refc6 tensor must be recomputed for correct
        results.
        """
        with torch.no_grad():
            allZ = torch.arange(self.Zmax)
            is_ = self.refsys[allZ, :]
            iz = self.zeff[is_]
            refh = self.refh[allZ, :] * F.softplus(self._scaleq)
            qref = iz
            qmod = iz + refh
            ones_like_qmod = torch.ones_like(qmod)
            qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
            alpha = (
                self.sscale[is_].view(-1, self.max_nref, 1)
                * self.secaiw[is_]
                * torch.where(
                    qmod > 1e-8,
                    torch.exp(
                        self.g_a
                        * (1 - torch.exp(self.gam[is_] * self.g_c * (1 - qref / qmod_)))
                    ),
                    math.exp(self.g_a) * ones_like_qmod,
                ).view(-1, self.max_nref, 1)
            )
            alpha = torch.max(
                self.ascale[allZ, :].view(-1, self.max_nref, 1)
                * (
                    self.alphaiw[allZ, :, :]
                    - self.hcount[allZ, :].view(-1, self.max_nref, 1) * alpha
                ),
                torch.zeros_like(alpha),
            )
            alpha_expanded = alpha.view(
                alpha.size(0), 1, alpha.size(1), 1, -1
            ) * alpha.view(1, alpha.size(0), 1, alpha.size(1), -1)
            self.register_buffer(
                "refc6",
                3.0
                / math.pi
                * torch.sum(
                    alpha_expanded * self.casimir_polder_weights.view(1, 1, 1, 1, -1),
                    -1,
                ),
                persistent=False,
            )

    def forward(
        self,
        N: int,
        Z: torch.Tensor,
        qa: torch.Tensor,
        rij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        compute_atomic_quantities: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx_i.numel() == 0:
            zeros = rij.new_zeros(N)
            return zeros, zeros, zeros
        # initialization of Zi/Zj and unit conversion
        rij = rij * self.convert2Bohr  # convert distances to Bohr
        Zi = Z[idx_i]
        Zj = Z[idx_j]

        # calculate coordination numbers
        rco = self.k2 * (self.rcov[Zi] + self.rcov[Zj])
        den = self.k4 * torch.exp(
            -((torch.abs(self.en[Zi] - self.en[Zj]) + self.k5) ** 2) / self.k6
        )
        tmp = den * 0.5 * (1.0 + torch.erf(-self.kn * (rij - rco) / rco))
        if self.cutoff is not None:
            tmp = tmp * switch_function(rij, self.cuton, self.cutoff)

        covcn = rij.new_zeros(N).index_add_(0, idx_i, tmp)

        # calculate gaussian weights
        gweights = torch.sum(
            self.ncount_mask[Z]
            * torch.exp(
                -self.wf
                * self.ncount_weight[Z]
                * (covcn.view(-1, 1, 1) - self.cn[Z]) ** 2
            ),
            -1,
        )
        norm = torch.sum(gweights, -1, True)
        # norm_ is used to prevent nans in backwards pass
        norm_ = torch.where(norm > 1e-8, norm, torch.ones_like(norm))
        gweights = torch.where(norm > 1e-8, gweights / norm_, self.fixgweights[Z])

        # calculate dispersion energy
        iz = self.zeff[Z].view(-1, 1)
        refq = self.refq[Z] * F.softplus(self._scaleq)
        qref = iz + refq
        qmod = iz + qa.view(-1, 1).expand(-1, self.refq.size(1))
        ones_like_qmod = torch.ones_like(qmod)
        qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
        zeta = (
            torch.where(
                qmod > 1e-8,
                torch.exp(
                    self.g_a
                    * (
                        1
                        - torch.exp(
                            self.gam[Z].view(-1, 1) * self.g_c * (1 - qref / qmod_)
                        )
                    )
                ),
                math.exp(self.g_a) * ones_like_qmod,
            )
            * gweights
        )
        if zeta.device.type == "cpu":  # indexing is faster on CPUs
            zetai = zeta[idx_i]
            zetaj = zeta[idx_j]
        else:  # gathering is faster on GPUs
            zetai = torch.gather(zeta, 0, idx_i.view(-1, 1).expand(-1, zeta.size(1)))
            zetaj = torch.gather(zeta, 0, idx_j.view(-1, 1).expand(-1, zeta.size(1)))
        refc6ij = self.refc6[Zi, Zj, :, :]
        zetaij = zetai.view(zetai.size(0), zetai.size(1), 1) * zetaj.view(
            zetaj.size(0), 1, zetaj.size(1)
        )
        c6ij = torch.sum((refc6ij * zetaij).view(refc6ij.size(0), -1), -1)
        sqrt_r4r2ij = math.sqrt(3) * self.sqrt_r4r2[Zi] * self.sqrt_r4r2[Zj]
        a1 = F.softplus(self._a1)
        a2 = F.softplus(self._a2)
        r0 = a1 * sqrt_r4r2ij + a2
        if self.cutoff is None:
            oor6 = 1 / (rij ** 6 + r0 ** 6)
            oor8 = 1 / (rij ** 8 + r0 ** 8)
        else:
            cut2 = self.cutoff ** 2
            cut6 = cut2 ** 3
            cut8 = cut2 * cut6
            tmp6 = r0 ** 6
            tmp8 = r0 ** 8
            cut6tmp6 = cut6 + tmp6
            cut8tmp8 = cut8 + tmp8
            tmpc = rij / self.cutoff - 1
            oor6 = (
                1 / (rij ** 6 + tmp6) - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * tmpc
            )
            oor8 = (
                1 / (rij ** 8 + tmp8) - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * tmpc
            )
            oor6 = torch.where(rij < self.cutoff, oor6, torch.zeros_like(oor6))
            oor8 = torch.where(rij < self.cutoff, oor8, torch.zeros_like(oor8))
        s6 = F.softplus(self._s6)
        s8 = F.softplus(self._s8)
        pairwise = -c6ij * (s6 * oor6 + s8 * sqrt_r4r2ij ** 2 * oor8) * self.convert2eV
        edisp = rij.new_zeros(N).index_add_(0, idx_i, pairwise)
        if compute_atomic_quantities:
            alpha = self.alpha[Z, :, 0]
            polarizabilities = torch.sum(zeta * alpha, -1) * self.convert2Angstrom3
            refc6ii = self.refc6[Z, Z, :, :]
            zetaii = zeta.view(zeta.size(0), zeta.size(1), 1) * zeta.view(
                zeta.size(0), 1, zeta.size(1)
            )
            c6_coefficients = (
                torch.sum((refc6ii * zetaii).view(refc6ii.size(0), -1), -1)
                * self.convert2eVAngstrom6
            )
        else:
            polarizabilities = rij.new_zeros(N)
            c6_coefficients = rij.new_zeros(N)
        return (edisp, polarizabilities, c6_coefficients)
