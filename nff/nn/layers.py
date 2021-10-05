from functools import partial
import sympy as sym
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
import numpy as np

from nff.utils import bessel_basis, real_sph_harm

zeros_initializer = partial(constant_, val=0.0)
DEFAULT_DROPOUT_RATE = 0.0
EPS = 1e-15


def gaussian_smearing(distances, offset, widths, centered=False):

    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances

    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))

    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    sample struct dictionary:

        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self,
                 start,
                 stop,
                 n_gaussians,
                 centered=False,
                 trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor(
            (offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        result = gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )

        return result


class Dense(nn.Linear):
    """ Applies a dense layer with activation: :math:`y = activation(Wx + b)`

    Args:
        in_features (int): number of input feature
        out_features (int): number of output features
        bias (bool): If set to False, the layer will not adapt the bias. (default: True)
        activation (callable): activation function (default: None)
        weight_init (callable): function that takes weight tensor and initializes (default: xavier)
        bias_init (callable): function that takes bias tensor and initializes (default: zeros initializer)
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """
        self.to(inputs.device)
        y = super().forward(inputs)

        # kept for compatibility with earlier versions of nff
        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y


def to_module(activation):
    from nff.utils.tools import layer_types
    return layer_types[activation]()


class PreActivation(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        activation,
        bias=True,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        if isinstance(activation, str):
            activation = to_module(activation)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """

        weights = self.weight.to(inputs.device)
        y = self.activation(inputs)
        y = torch.einsum('ij,kj->ki', weights, y)

        if self.bias is not None:
            y = y + self.bias.to(y.device)

        if hasattr(self, "dropout"):
            y = self.dropout(y)

        return y


class BatchedPreActivation(nn.Conv1d):
    """
    Pre-activation layer that can convert an input to multiple different
    outputs in parallel. This is equivalent to generating N preactivation 
    layers and applying them in series to the input to generate N outputs, 
    but done in parallel instead.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_out,
        activation,
        bias=True,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_channels=num_out,
                         out_channels=(num_out * out_features),
                         kernel_size=in_features,
                         groups=num_out,
                         bias=bias)

        # separate activations in case they're learnable
        if activation is None:
            self.activations = None
        else:
            self.activations = nn.ModuleList([to_module(activation)
                                              for _ in range(num_out)])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_out = num_out

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Output of the dense layer.
        """

        if self.activations is None:
            y = inputs.clone()
        else:
            y = torch.stack([act(i) for i, act in
                             zip(inputs, self.activations)])

        # switch ordering from (num_channels, num_samples, F)
        # to (num_samples, num_channels, F)
        y_transpose = y.transpose(0, 1)
        y = super().forward(y_transpose)

        if hasattr(self, "dropout"):
            y = self.dropout(y)

        # switch from (num_samples, F x num_channels)
        # to (num_channels, num_samples, F)
        num_channels, num_samples, feat_dim = inputs.shape

        y = (y.reshape(num_samples, num_channels, feat_dim)
             .transpose(0, 1))

        # channel = 2
        # sample = 7
        # if self.activations:
        #     test_inp = self.activations[2](inputs[channel, sample])
        # else:
        #     test_inp = inputs[channel, sample]
        # test = torch.matmul(self.weight[channel * 128:
        #                                 (channel + 1) * 128, 0, :],
        #                     test_inp) + self.bias[channel * 128:
        #                                           (channel + 1) * 128]
        # import pdb
        # pdb.set_trace()

        # print(abs(y[channel, sample] - test).mean())

        return y


class Envelope(nn.Module):
    """
    Layer for adding a polynomial envelope to the spherical and
    radial Bessel functions in DimeNet.
    """

    def __init__(self, p):
        """
        Args:
            p (int): exponent in the damping envelope
        Returns:
            None
        """
        super().__init__()
        self.p = p

    def forward(self, d):
        """
        Args:
            d (torch.Tensor): tensor of distances
        Returns:
            u (torch.Tensor): polynomial of the distances
        """
        p = self.p
        u = 1 - (p + 1) * (p + 2) / 2 * d ** p \
            + p * (p + 2) * d ** (p + 1) \
            - p * (p + 1) / 2 * d ** (p + 2)
        return u


class DimeNetSphericalBasis(nn.Module):
    """
    Spherical basis layer for DimeNet.
    """

    def __init__(self,
                 l_spher,
                 n_spher,
                 cutoff,
                 envelope_p):
        """
        Args:
            l_spher (int): maximum l value in the spherical
                basis functions
            n_spher (int): maximum n value in the spherical
                basis functions
            cutoff (float): cutoff distance in the neighbor list
            envelope_p (int): exponent in the damping envelope
        Returns:
            None
        """

        super().__init__()

        assert n_spher <= 64

        self.n_spher = n_spher
        self.l_spher = l_spher
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_p)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(l_spher, n_spher)
        self.sph_harm_formulas = real_sph_harm(l_spher)
        self.sph_funcs = []
        self.bessel_funcs = []

        # create differentiable Torch functions through
        # sym.lambdify

        x = sym.symbols('x')
        theta = sym.symbols('theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(l_spher):
            if i == 0:
                first_sph = sym.lambdify(
                    [theta], self.sph_harm_formulas[i][0], modules)(0)
                self.sph_funcs.append(
                    lambda tensor: torch.zeros_like(tensor) + first_sph)
            else:
                self.sph_funcs.append(sym.lambdify(
                    [theta], self.sph_harm_formulas[i][0], modules))
            for j in range(n_spher):
                self.bessel_funcs.append(sym.lambdify(
                    [x], self.bessel_formulas[i][j], modules))

    def forward(self, d, angles, kj_idx):
        """
        Args:
            d (torch.Tensor): tensor of distances
            angles (torch.Tensor): tensor of angles
            kj_idx (torch.LongTensor): nbr_list indices corresponding
                to the k,j indices in the angle list.
        """

        # compute the radial functions with arguments d / cutoff
        d_scaled = d / self.cutoff
        rbf = [f(d_scaled) for f in self.bessel_funcs]
        rbf = torch.stack(rbf, dim=1)

        # multiply the radial basis functions by the envelope
        u = self.envelope(d_scaled)
        rbf_env = u[:, None] * rbf

        # we want d_kj for each angle alpha_{kj, ji}
        # = angle_{ijk}, so we want to order the distances
        # so they align with the kj indices of `angles`

        rbf_env = rbf_env[kj_idx.long()]
        rbf_env = rbf_env.reshape(*torch.tensor(
            rbf_env.shape[:2]).tolist())

        # get the angular functions
        cbf = [f(angles) for f in self.sph_funcs]
        cbf = torch.stack(cbf, dim=1)
        # repeat for n_spher
        cbf = cbf.repeat_interleave(self.n_spher, dim=1)

        # multiply with rbf and return

        return rbf_env * cbf


class DimeNetRadialBasis(nn.Module):
    """
    Radial basis layer for DimeNet.
    """

    def __init__(self,
                 n_rbf,
                 cutoff,
                 envelope_p):
        """
        Args:
            n_rbf (int): number of radial basis functions
            cutoff (float): cutoff distance in the neighbor list
            envelope_p (int): exponent in the damping envelope
        Returns:
            None
        """

        super().__init__()
        n = torch.arange(1, n_rbf + 1).float()
        # initialize k_n but let it be learnable
        self.k_n = nn.Parameter(n * np.pi / cutoff)
        self.envelope = Envelope(envelope_p)
        self.cutoff = cutoff

    def forward(self, d):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """
        pref = (2 / self.cutoff) ** 0.5
        arg = torch.sin(self.k_n * d) / d
        u = self.envelope(d / self.cutoff)

        return pref * arg * u


class Diagonalize(nn.Module):

    def __init__(self):
        super().__init__()

    def _generate_inputs(self, d_mat):
        d0 = d_mat[:, 0, 0]
        d1 = d_mat[:, 1, 1]
        lam = d_mat[:, 0, 1]

        return d0, d1, lam

    def compute_v(self, d0, d1, lam, e0, state):

        term_1 = -d0 + d1
        term_2 = (d0 ** 2 - 2 * d0 * d1 + d1 ** 2 + 4 * lam ** 2 + EPS) ** 0.5
        denom = 2 * lam

        if state == 'lower':
            v_element_0 = -(term_1 + term_2) / denom
        elif state == 'upper':
            v_element_0 = -(term_1 - term_2) / denom

        v_element_0 = v_element_0.reshape(*e0.shape)
        v_crude = torch.stack([v_element_0,
                               torch.ones_like(v_element_0)],
                              dim=-1)
        v_norm = torch.norm(v_crude, dim=-1).reshape(-1, 1)
        v = v_crude / v_norm

        return v

    def compute_U(self, d0, d1, lam, e0):

        v_list = []
        for state in ['lower', 'upper']:
            v = self.compute_v(d0=d0,
                               d1=d1,
                               lam=lam,
                               e0=e0,
                               state=state)
            v_list.append(v)

        U_inv = torch.cat([v_list[0], v_list[1]], dim=-1).reshape(-1, 2, 2)
        U = U_inv.transpose(1, 2)

        return U

    def forward(self,
                d_mat):

        d0, d1, lam = self._generate_inputs(d_mat)
        e0 = 1 / 2 * (d0 + d1 - ((d0 - d1) ** 2 + 4 * lam ** 2 + EPS) ** 0.5)
        e1 = 1 / 2 * (d0 + d1 + ((d0 - d1) ** 2 + 4 * lam ** 2 + EPS) ** 0.5)

        eigs = torch.stack([e0, e1], dim=-1)
        U = self.compute_U(d0=d0,
                           d1=d1,
                           lam=lam,
                           e0=e0)

        return eigs, U


class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):

        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output


class PainnRadialBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 learnable_k):
        super().__init__()

        self.n = torch.arange(1, n_rbf + 1).float()
        if learnable_k:
            self.n = nn.Parameter(self.n)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function

        denom = torch.where(shape_d == 0,
                            torch.tensor(1.0, device=device),
                            shape_d)
        num = torch.where(shape_d == 0,
                          coef,
                          torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff,
                             torch.tensor(0.0, device=device),
                             num / denom)

        return output


class ExpNormalBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 learnable_mu,
                 learnable_beta):
        super().__init__()

        self.mu = torch.linspace(np.exp(-cutoff), 1, n_rbf)

        init_beta = (2 / n_rbf * (1 - np.exp(-cutoff))) ** (-2)
        self.beta = (torch.ones_like(self.mu) * init_beta)

        if learnable_mu:
            self.mu = nn.Parameter(self.mu)
        if learnable_beta:
            self.beta = nn.Parameter(self.beta)

        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        mu = self.mu.to(dist.device)
        beta = self.beta.to(dist.device)

        arg = beta * (torch.exp(-shape_d) - mu) ** 2
        output = torch.exp(-arg)

        return output


class StochasticIncrease(nn.Module):
    """
    Module that stochastically enhances the magnitude of a network
    output. Designed with energy gaps in mind so that small gaps
    are stochastically increased while large ones are basically
    unchanged. This biases the model toward producing small gaps
    so that it can account for the stochastic increases.
    """

    def __init__(self,
                 exp_coef,
                 order,
                 rate):

        super().__init__()
        self.exp_coef = exp_coef
        self.order = order
        self.rate = rate

    def forward(self, output):

        rnd = np.random.rand()
        do_increase = rnd < self.rate
        if do_increase:
            arg = -self.exp_coef * (output.abs() ** self.order)
            new_output = output * (1 + torch.exp(arg))
        else:
            new_output = output
        return new_output


class Gaussian(nn.Module):
    def __init__(self,
                 mean,
                 sigma,
                 learnable_mean,
                 learnable_sigma,
                 normalize):

        super().__init__()

        self.mean = mean
        self.sigma = sigma
        self.normalize = normalize

        if learnable_mean:
            self.mean = torch.nn.Parameter(torch.Tensor([self.mean]))
        if learnable_sigma:
            self.sigma = torch.nn.Parameter(torch.Tensor([self.sigma]))

    def forward(self, inp):
        out = torch.exp(-(inp - self.mean) ** 2 / (2 * self.sigma ** 2))
        if self.normalize:
            denom = self.sigma * (2 * np.pi) ** 0.5
            out = out / denom
        return out
