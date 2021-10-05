"""
Special functions for DimeNet and SpookyNet.
Dimenet functions taken directly from 
https://github.com/klicperajo/
dimenet/blob/master/dimenet/model/
layers/basis_utils.py.
"""

import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import sympy as sym
import copy
import torch
import math


EPS = 1e-15


# DimeNet

def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')

    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(
                                                x, zeros[order, i]*x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """
    Computes the constant pre-factor for the spherical harmonic of degree l and order m
    input:
    l: int, l>=0
    m: int, -l<=m<=l
    """
    return ((2*l+1) * np.math.factorial(l-abs(m))
            / (4*np.pi*np.math.factorial(l+abs(m))))**0.5


def associated_legendre_polynomials(l, zero_m_only=True):
    """
    Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
    """
    z = sym.symbols('z')
    P_l_m = [[0]*(j+1) for j in range(l)]

    P_l_m[0][0] = 1
    if l > 0:
        P_l_m[1][0] = z

        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2*j-1)*z*P_l_m[j-1][0] - (j-1)*P_l_m[j-2][0])/j)
        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1-2*i)*P_l_m[i-1][i-1])
                if i + 1 < l:
                    P_l_m[i+1][i] = sym.simplify((2*i+1)*z*P_l_m[i][i])
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        ((2*j-1) * z * P_l_m[j-1][i]
                            - (i+j-1) * P_l_m[j-2][i]) / (j - i))

    return P_l_m


def real_sph_harm(l,
                  zero_m_only=True,
                  spherical_coordinates=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, l):
            x = sym.symbols('x')
            y = sym.symbols('y')
            S_m += [x*S_m[i-1] + y*C_m[i-1]]
            C_m += [x*C_m[i-1] - y*S_m[i-1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)
    if spherical_coordinates:
        theta = sym.symbols('theta')
        z = sym.symbols('z')
        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))
        if not zero_m_only:
            phi = sym.symbols('phi')
            for i in range(len(S_m)):
                S_m[i] = S_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))
            for i in range(len(C_m)):
                C_m[i] = C_m[i].subs(x, sym.sin(
                    theta)*sym.cos(phi)).subs(y, sym.sin(theta)*sym.sin(phi))

    Y_func_l_m = [['0']*(2*j + 1) for j in range(l)]
    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j])
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2**0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j])

    return Y_func_l_m


# SpookyNet

def A_m(x, y, m):
    device = x.device
    p_vals = torch.arange(0, m + 1,
                          device=device)
    q_vals = m - p_vals
    x_p = x.reshape(-1, 1) ** p_vals
    y_q = y.reshape(-1, 1) ** q_vals
    sin = torch.sin(np.pi / 2 * (m - p_vals))
    binoms = (torch.Tensor([sp.binom(m, int(p))
                            for p in p_vals])
              .to(device))
    out = (binoms * x_p * y_q * sin).sum(-1)

    return out


def B_m(x, y, m):
    device = x.device
    p_vals = torch.arange(0, m + 1,
                          device=device)
    q_vals = m - p_vals
    x_p = x.reshape(-1, 1) ** p_vals
    y_q = y.reshape(-1, 1) ** q_vals
    cos = torch.cos(np.pi / 2 * (m - p_vals))
    binoms = (torch.Tensor([sp.binom(m, int(p)) for p in p_vals])
              .to(device))
    out = (binoms * x_p * y_q * cos).sum(-1)

    return out


def c_plm(p, l, m):
    terms = [(-1) ** p,
             1 / (2 ** l),
             sp.binom(l, p),
             sp.binom(2 * l - 2 * p, l),
             sp.factorial(l - 2 * p),
             1 / sp.factorial(l - 2 * p - m)]
    out = torch.Tensor(terms).prod()
    return out


def make_c_table(l_max):
    c_table = {}
    for l in range(l_max + 1):
        for m in range(-l, l+1):
            for p in range(0, math.floor((l - m) / 2) + 1):
                c_table[(p, l, m)] = c_plm(p, l, m)
    return c_table


def pi_l_m(r,
           z,
           l,
           m,
           c_table):

    device = r.device
    pref = (sp.factorial(l - m) / sp.factorial(l + m)) ** 0.5
    p_vals = (torch.arange(0, math.floor((l - m) / 2) + 1,
                           device=device,
                           dtype=torch.float))

    c_coefs = (torch.Tensor([c_table[(int(p), l, m)]
                             for p in p_vals])
               .to(device))
    r_p = r.reshape(-1, 1) ** (2 * p_vals - l)
    z_q = z.reshape(-1, 1) ** (l - 2 * p_vals - m)

    out = pref * (c_coefs * r_p * z_q).sum(-1)

    return out


def norm(vec):
    result = ((vec ** 2 + EPS).sum(-1)) ** 0.5
    return result


def y_lm(r_ij,
         r,
         l,
         m,
         c_table):

    x = r_ij[:, 0].reshape(-1, 1)
    y = r_ij[:, 1].reshape(-1, 1)
    z = r_ij[:, 2].reshape(-1, 1)

    pi = pi_l_m(r=r,
                z=z,
                l=l,
                m=abs(m),
                c_table=c_table)

    if m < 0:
        a = A_m(x, y, abs(m))
        out = (2 ** 0.5) * pi * a
    elif m == 0:
        out = pi
    elif m > 0:
        b = B_m(x, y, abs(m))
        out = (2 ** 0.5) * pi * b

    return out


def make_y_lm(l_max):
    c_table = make_c_table(l_max)

    def func(r_ij, r, l, m):
        out = y_lm(r_ij=r_ij,
                   r=r,
                   l=l,
                   m=m,
                   c_table=c_table)

        return out
    return func


def spooky_f_cut(r, r_cut):
    arg = r ** 2 / ((r_cut - r) * (r_cut + r))
    # arg < 20 is for numerical stability
    # Anything > 20 will give under 1e-9
    output = torch.where(
        (r < r_cut) * (arg < 20),
        torch.exp(-arg),
        torch.Tensor([0]).to(r.device)
    )

    return output


def b_k(x,
        bern_k):
    device = x.device
    k_vals = (torch.arange(0, bern_k, device=device)
              .to(torch.float))
    binoms = (torch.Tensor([sp.binom(bern_k - 1, int(k))
                            for k in k_vals])
              .to(device))
    out = binoms * (x ** k_vals) * (1-x) ** (bern_k - 1 - k_vals)
    return out


def rho_k(r,
          r_cut,
          bern_k,
          gamma):

    arg = torch.exp(-gamma * r)
    out = b_k(arg, bern_k) * spooky_f_cut(r, r_cut)

    return out


def get_g_func(l,
               r_cut,
               bern_k,
               gamma,
               y_lm_fn):

    def fn(r_ij):

        r = norm(r_ij).reshape(-1, 1)
        n_pairs = r_ij.shape[0]
        device = r_ij.device

        m_vals = list(range(-l, l + 1))
        y = torch.stack([y_lm_fn(r_ij, r, l, m) for m in
                         m_vals]).transpose(0, 1)
        rho = rho_k(r, r_cut, bern_k, gamma)
        g = torch.ones(n_pairs,
                       bern_k,
                       len(m_vals),
                       device=device)
        g = g * rho.reshape(n_pairs, -1, 1)
        g = g * y.reshape(n_pairs, 1, -1)

        return g

    return fn


def make_g_funcs(bern_k,
                 gamma,
                 r_cut,
                 l_max=2):
    y_lm_fn = make_y_lm(l_max)
    g_funcs = {}
    letters = {0: "s", 1: "p", 2: "d"}

    for l in range(0, l_max + 1):

        letter = letters[l]
        name = f"g_{letter}"
        g_func = get_g_func(l=l,
                            r_cut=r_cut,
                            bern_k=bern_k,
                            gamma=gamma,
                            y_lm_fn=y_lm_fn)
        g_funcs[name] = copy.deepcopy(g_func)

    return g_funcs
