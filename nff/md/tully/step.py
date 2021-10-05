"""
Functions for performing a Tully time step
"""

import copy
from functools import partial

import numpy as np
import torch


def compute_T(nacv,
              vel,
              c):

    # vel has shape num_samples x num_atoms x 3
    # nacv has shape num_samples x num_states x num_states
    # x num_atoms x 3
    # T has shape num_samples x (num_states x num_states)

    T = (vel.reshape(vel.shape[0], 1, 1, -1, 3)
         * nacv).sum((-1, -2))

    # anything that's nan has too big a gap
    # for hopping and should therefore have T=0
    T[np.isnan(T)] = 0

    num_states = nacv.shape[1]
    coupling = np.einsum('nij, nj-> ni', T, c[:, :num_states])

    return T, coupling


def get_dc_dt(c,
              vel,
              nacv,
              energy,
              hbar=1):

    # energies have shape num_samples x num_states
    w = energy / hbar

    # T has dimension num_samples x (num_states x num_states)
    T, coupling = compute_T(nacv=nacv,
                            vel=vel,
                            c=c)

    dc_dt = -(1j * w * c + coupling)

    return dc_dt, T


def get_a(c):
    # a has dimension num_samples x (num_states x num_states)
    num_samples = c.shape[0]
    num_states = c.shape[1]

    a = np.zeros((num_samples, num_states, num_states),
                 dtype='complex128')
    for i in range(num_states):
        for j in range(num_states):
            a[..., i, j] = (np.conj(c[..., i])
                            * c[..., j])
    return a


def remove_self_hop(p,
                    surfs):

    same_surfs = surfs.reshape(-1, 1)
    np.put_along_axis(p,
                      same_surfs,
                      np.zeros_like(same_surfs),
                      axis=-1)

    return p


def get_tully_p(c,
                T,
                dt,
                surfs,
                num_adiabat,
                **kwargs):
    """
    Tully surface hopping probability
    """

    a = get_a(c)[:, :num_adiabat, :num_adiabat]

    # T, a and b have dimension
    # num_samples x (num_states x num_states).
    # The diagonals of T (and hence b) are zero.

    b = -2 * np.real(np.conj(a) * T)

    # a_surf has dimension num_samples x 1
    a_surf = np.stack([sample_a[surf, surf] for
                       sample_a, surf in zip(a, surfs)]).reshape(-1, 1)

    # b_surf has dimension num_samples x num_states
    b_surf = np.stack([sample_b[:, surf] for
                       sample_b, surf in zip(b, surfs)])

    # p has dimension num_samples x num_states, for the
    # hopping probability of each sample to all other
    # states

    # p is real anyway - taking the real part just gets rid of
    # the +0j parts

    p = np.real(dt * b_surf / a_surf)

    # no hopping from current state to self
    p = remove_self_hop(p=p,
                        surfs=surfs)

    # only hop among adiabatic states of interest
    p = p[:, :num_adiabat]

    return p


def get_sharc_p(old_c,
                new_c,
                P,
                surfs,
                num_adiabat,
                **kwargs):
    """
    P is the propagator.
    """

    num_samples = old_c.shape[0]
    num_states = old_c.shape[1]

    other_surfs = get_other_surfs(surfs=surfs,
                                  num_states=num_states,
                                  num_samples=num_samples)

    c_beta_t = np.take_along_axis(old_c,
                                  surfs.reshape(-1, 1),
                                  axis=-1)
    c_beta_dt = np.take_along_axis(new_c,
                                   surfs.reshape(-1, 1),
                                   axis=-1)

    c_alpha_dt = np.take_along_axis(new_c,
                                    other_surfs,
                                    axis=-1)

    # `P` has dimension num_samples x num_states x num_states

    P_alpha_beta = np.take_along_axis(np.take_along_axis(
        P,
        surfs.reshape(-1, 1, 1),
        axis=-1).squeeze(-1),
        other_surfs,
        axis=-1
    )

    P_beta_beta = np.take_along_axis(np.take_along_axis(
        P,
        surfs.reshape(-1, 1, 1),
        axis=-1).squeeze(-1),
        surfs.reshape(-1, 1),
        axis=-1
    )

    # h_alpha is the transition probability from the current state
    # to alpha

    num = np.real(c_alpha_dt * np.conj(P_alpha_beta) * np.conj(c_beta_t))
    denom = abs(c_beta_t) ** 2 - np.real(c_beta_dt * np.conj(P_beta_beta)
                                         * np.conj(c_beta_t))
    pref = 1 - abs(c_beta_dt) ** 2 / abs(c_beta_t) ** 2

    h = np.zeros((num_samples, num_states))
    np.put_along_axis(h,
                      other_surfs,
                      pref * num / denom,
                      axis=-1)
    h[h < 0] = 0

    # only hop among adiabatic states of interest
    h = h[:, :num_adiabat]

    return h


def get_p_hop(hop_eqn='sharc',
              **kwargs):

    if hop_eqn == 'sharc':
        p = get_sharc_p(**kwargs)
    elif hop_eqn == 'tully':
        p = get_tully_p(**kwargs)
    else:
        raise NotImplementedError

    return p


def get_new_surf(p_hop,
                 surfs,
                 max_gap_hop,
                 energy):

    num_samples = p_hop.shape[0]
    lhs = np.concatenate([np.zeros(num_samples).reshape(-1, 1),
                          p_hop.cumsum(axis=-1)],
                         axis=-1)[:, :-1]
    rhs = lhs + p_hop
    r = np.random.rand(num_samples).reshape(-1, 1)
    hop = (lhs < r) * (r <= rhs)
    hop_idx = np.stack(hop.nonzero(), axis=-1)

    new_surfs = copy.deepcopy(surfs)
    new_surfs[hop_idx[:, 0]] = hop_idx[:, 1]

    if max_gap_hop is None:
        return new_surfs

    old_en = np.take_along_axis(energy,
                                surfs.reshape(-1, 1),
                                axis=-1).squeeze(-1)
    new_en = np.take_along_axis(energy,
                                new_surfs.reshape(-1, 1),
                                axis=-1).squeeze(-1)
    gaps = abs(old_en - new_en)
    bad_idx = gaps >= max_gap_hop
    new_surfs[bad_idx] = surfs[bad_idx]

    return new_surfs


def solve_quadratic(vel,
                    nac_dir,
                    old_en,
                    new_en,
                    mass):
    a = (1 / (2 * mass.reshape(1, -1, 1))
         * nac_dir ** 2).sum((-1, -2)).astype('complex128')
    b = (vel * nac_dir).sum((-1, -2)).astype('complex128')
    c = (new_en - old_en).astype('complex128')

    sqrt = np.sqrt(b ** 2 - 4 * a * c)
    scale_pos = (-b + sqrt) / (2 * a)
    scale_neg = (-b - sqrt) / (2 * a)

    # take solution with smallest absolute value of
    # scaling factor
    scales = np.concatenate([scale_pos.reshape(-1, 1),
                             scale_neg.reshape(-1, 1)],
                            axis=1)
    scale_argmin = np.argmin(abs(scales), axis=1)
    scale = np.take_along_axis(scales,
                               scale_argmin.reshape(-1, 1),
                               axis=1)

    scale[np.imag(scale) != 0] = np.nan
    scale = np.real(scale)

    return scale


def get_simple_scale(mass,
                     new_en,
                     old_en,
                     vel):

    m = mass.reshape(1, -1, 1)
    gap = old_en - new_en
    arg = ((2 * gap + (m * vel ** 2).sum((-1, -2)))
           .astype('complex128'))
    num = np.sqrt(arg)
    denom = np.sqrt((m * vel ** 2).sum((-1, -2)))

    v_scale = num / denom

    # reset frustrated hops
    v_scale[np.imag(v_scale) != 0] = np.nan
    v_scale = np.real(v_scale)

    return v_scale


def rescale(energy,
            vel,
            nacv,
            mass,
            surfs,
            new_surfs,
            simple_scale):
    """
    Velocity re-scaling, from:

    Landry, B.R. and Subotnik, J.E., 2012. How to recover Marcus theory with 
    fewest switches surface hopping: Add just a touch of decoherence. The 
    Journal of chemical physics, 137(22), p.22A513.

    If no NACV is available, the KE is simply rescaled to conserve energy.
    This is the default in SHARC.
    """

    # old and new energies
    old_en = np.take_along_axis(energy, surfs.reshape(-1, 1),
                                -1).reshape(-1)
    new_en = np.take_along_axis(energy, new_surfs.reshape(-1, 1),
                                -1).reshape(-1)

    if simple_scale or nacv is None:
        v_scale = get_simple_scale(mass=mass,
                                   new_en=new_en,
                                   old_en=old_en,
                                   vel=vel)
        new_vel = v_scale.reshape(-1, 1, 1) * vel
        return new_vel

    # nacvs connecting old to new surfaces
    ones = [1] * 4
    start_nacv = np.take_along_axis(nacv, surfs
                                    .reshape(-1, *ones),
                                    axis=1)
    pair_nacv = np.take_along_axis(start_nacv, new_surfs
                                   .reshape(-1, *ones),
                                   axis=2
                                   ).squeeze(1).squeeze(1)

    # nacv unit vector
    norm = np.linalg.norm(pair_nacv, axis=-1)
    # for anything that didn't hop
    norm[norm == 0] = np.nan
    nac_dir = pair_nacv / norm.reshape(*pair_nacv.shape[:-1], 1)

    # solve quadratic equation for momentum rescaling
    scale = solve_quadratic(vel=vel,
                            nac_dir=nac_dir,
                            old_en=old_en,
                            new_en=new_en,
                            mass=mass)

    # scale the velocity
    new_vel = (scale.reshape(-1, 1, 1) * nac_dir
               / mass.reshape(1, -1, 1)
               + vel)

    return new_vel


def try_hop(c,
            p_hop,
            surfs,
            vel,
            nacv,
            mass,
            energy,
            max_gap_hop,
            simple_scale):
    """
    `energy` has dimension num_samples x num_states
    """

    new_surfs = get_new_surf(p_hop=p_hop,
                             surfs=surfs,
                             max_gap_hop=max_gap_hop,
                             energy=energy)

    new_vel = rescale(energy=energy,
                      vel=vel,
                      nacv=nacv,
                      mass=mass,
                      surfs=surfs,
                      new_surfs=new_surfs,
                      simple_scale=simple_scale)

    # reset any frustrated hops or things that didn't hop
    frustrated = np.isnan(new_vel).any((-1, -2)).nonzero()[0]
    new_vel[frustrated] = vel[frustrated]
    new_surfs[frustrated] = surfs[frustrated]

    return new_surfs, new_vel


def runge_c(c,
            vel,
            nacv,
            energy,
            elec_dt,
            hbar=1):
    """
    Runge-Kutta step for c
    """

    deriv = partial(get_dc_dt,
                    vel=vel,
                    nacv=nacv,
                    energy=energy,
                    hbar=hbar)

    k1, T1 = deriv(c)
    k2, T2 = deriv(c + elec_dt * k1 / 2)
    k3, T3 = deriv(c + elec_dt * k2 / 2)
    k4, T4 = deriv(c + elec_dt * k3)

    new_c = c + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return new_c, T1


def remove_T_nan(T, S):
    num_states = S.shape[1]
    nan_idx = np.bitwise_not(np.isfinite(T))

    num_nan = int(nan_idx.nonzero()[0].reshape(-1).shape[0]
                  / num_states ** 2)
    eye = (np.eye(num_states).reshape(-1, num_states, num_states)
           .repeat(num_nan, axis=0)).reshape(-1)
    T[nan_idx] = eye

    return T


def get_implicit_diabat(c,
                        elec_substeps,
                        old_H_ad,
                        new_H_ad,
                        new_U,
                        old_U,
                        dt,
                        hbar=1):

    num_ad = c.shape[1]
    S = np.einsum('...ki, ...kj -> ...ij',
                  old_U, new_U)[:, :num_ad, :num_ad]

    s_t_s = np.einsum('...ji, ...jk -> ...ik', S, S)
    lam, O = np.linalg.eigh(s_t_s)

    # in case any eigenvalues are 0 or slightly negative
    with np.errstate(divide='ignore', invalid='ignore'):
        lam_half = np.stack([np.diag(i ** (-1 / 2))
                             for i in lam])
    T = np.einsum('...ij, ...jk, ...kl, ...ml -> ...im',
                  S, O, lam_half, O)

    # set T to the identity for any cases in which one of
    # the eigenvalues is 0
    T = remove_T_nan(T=T, S=S)

    T_inv = T.transpose(0, 2, 1)

    old_H_d = old_H_ad
    new_H_d = np.einsum('...ij, ...jk, ...lk -> ...il',
                        T, new_H_ad, T)

    return old_H_d, new_H_d, T_inv


def adiabatic_c(c,
                elec_substeps,
                old_H_plus_nacv,
                new_H_plus_nacv,
                dt,
                hbar=1,
                **kwargs):

    num_samples = old_H_plus_nacv.shape[0]
    num_states = old_H_plus_nacv.shape[1]
    n = elec_substeps

    exp = (np.eye(num_states, num_states)
           .reshape(1, num_states, num_states)
           .repeat(num_samples, axis=0))

    delta_tau = dt / n

    for i in range(1, n + 1):
        new_exp = torch.tensor(
            -1j / hbar * (
                old_H_plus_nacv + i / n *
                (new_H_plus_nacv - old_H_plus_nacv)
            )
            * delta_tau
        ).matrix_exp().numpy()
        exp = np.einsum('ijk, ikl -> ijl', exp, new_exp)

    P = exp
    c_new = np.einsum('ijk, ik -> ij', P, c)

    return c_new, P


def diabatic_c(c,
               elec_substeps,
               new_U,
               old_U,
               dt,
               explicit_diabat,
               hbar=1,
               old_H_d=None,
               new_H_d=None,
               old_H_ad=None,
               new_H_ad=None,
               **kwargs):

    if not explicit_diabat:
        old_H_d, new_H_d, T_inv = get_implicit_diabat(
            c=c,
            elec_substeps=elec_substeps,
            old_H_ad=old_H_ad,
            new_H_ad=new_H_ad,
            new_U=new_U,
            old_U=old_U,
            dt=dt,
            hbar=hbar)

    num_samples = old_H_d.shape[0]
    num_states = old_H_d.shape[1]
    n = elec_substeps

    exp = (np.eye(num_states, num_states)
           .reshape(1, num_states, num_states)
           .repeat(num_samples, axis=0))

    delta_tau = dt / n

    for i in range(1, n + 1):

        new_exp = torch.tensor(
            -1j / hbar * (
                old_H_d + i / n * (new_H_d - old_H_d)
            )
            * delta_tau
        ).matrix_exp().numpy()
        exp = np.einsum('ijk, ikl -> ijl', exp, new_exp)

    if explicit_diabat:
        # new_U has dimension num_samples x num_states x num_states
        T = old_U
        T_inv = new_U.transpose(0, 2, 1)
        P = np.einsum('ijk, ikl, ilm -> ijm',
                      T_inv, exp, T)
    else:
        # if implicit, T(t) = identity
        P = np.einsum('ijk, ikl -> ijl',
                      T_inv, exp)

    c_new = np.einsum('ijk, ik -> ij', P, c)

    # print(abs(c_new[30]) ** 2)

    return c_new, P


def verlet_step_1(forces,
                  surfs,
                  vel,
                  xyz,
                  mass,
                  dt):

    # `forces` has dimension (num_samples x num_states
    # x num_atoms x 3)
    # `surfs` has dimension `num_samples`

    surf_forces = np.take_along_axis(
        forces, surfs.reshape(-1, 1, 1, 1),
        axis=1
    ).squeeze(1)

    # `surf_forces` has dimension (num_samples x
    #  num_atoms x 3)
    # `mass` has dimension `num_atoms`
    accel = surf_forces / mass.reshape(1, -1, 1)

    # `vel` and `xyz` each have dimension
    # (num_samples x num_atoms x 3)

    new_xyz = xyz + vel * dt + 0.5 * accel * dt ** 2
    new_vel = vel + 0.5 * dt * accel

    return new_xyz, new_vel


def verlet_step_2(forces,
                  surfs,
                  vel,
                  mass,
                  dt):

    surf_forces = np.take_along_axis(
        forces, surfs.reshape(-1, 1, 1, 1),
        axis=1
    ).squeeze(1)

    accel = surf_forces / mass.reshape(1, -1, 1)
    new_vel = vel + 0.5 * dt * accel

    return new_vel


def delta_F_for_tau(forces):

    num_samples = forces.shape[0]
    num_states = forces.shape[1]
    num_atoms = forces.shape[-2]

    delta_F = np.zeros((num_samples, num_states, num_states, num_atoms, 3))
    delta_F += forces.reshape(num_samples, num_states, 1, num_atoms, 3)
    delta_F -= forces.reshape(num_samples, 1, num_states, num_atoms, 3)

    return delta_F


def get_diag_delta_R(delta_R):

    num_states = delta_R.shape[1]
    diag_delta_R = np.take_along_axis(delta_R, np.arange(num_states)
                                      .reshape(1, -1, 1, 1, 1), axis=2
                                      ).repeat(num_states, axis=2)
    return diag_delta_R


def get_tau_d(forces,
              energy,
              force_nacv,
              delta_R,
              hbar=1,
              zeta=1):

    # tau_d^{ni} has shape num_samples x num_states x num_states
    # delta_R, delta_P, and force_nacv have shape
    # num_samples x num_states x num_states x num_atoms x 3

    delta_F = delta_F_for_tau(forces)
    diag_delta_R = get_diag_delta_R(delta_R)

    term_1 = (delta_F * diag_delta_R).sum((-1, -2)) / (2 * hbar)
    term_2 = - 2 * abs(zeta / hbar * (
        force_nacv.transpose((0, 2, 1, 3, 4))
        * diag_delta_R)
        .sum((-1, -2))
    )

    tau = term_1 + term_2

    return tau


def get_tau_reset(forces,
                  energy,
                  force_nacv,
                  delta_R,
                  hbar=1,
                  zeta=1):

    delta_F = delta_F_for_tau(forces)
    diag_delta_R = get_diag_delta_R(delta_R)

    tau_reset = -(delta_F * diag_delta_R).sum((-1, -2)) / (2 * hbar)

    return tau_reset


def matmul(a, b):
    """
    Matrix multiplication in electronic subspace
    """

    out = np.einsum('ijk..., ikl...-> ijl...', a, b)
    return out


def commute(a, b):
    """
    Commute two operators
    """

    comm = matmul(a, b) - matmul(b, a)

    return comm


def get_term_3(nacv,
               delta,
               vel):
    num_samples = nacv.shape[0]
    num_states = nacv.shape[1]
    num_atoms = nacv.shape[3]

    d_beta = np.zeros((num_samples, num_states, num_states,
                       3 * num_atoms, 3 * num_atoms))

    d_beta += nacv.reshape(num_samples, num_states,
                           num_states, 1, 3 * num_atoms,)

    delta_R_alpha = np.zeros((num_samples,
                              num_states,
                              num_states,
                              3 * num_atoms,
                              3 * num_atoms),
                             dtype='complex128')

    delta_R_alpha += delta.reshape(num_samples,
                                   num_states,
                                   num_states,
                                   3 * num_atoms, 1)

    vel_reshape = vel.reshape(num_samples, 1, 1, 1, 3 * num_atoms)
    term_3 = -(commute(d_beta, delta_R_alpha)
               * vel_reshape
               ).sum(-1)
    term_3 = term_3.reshape(num_samples,
                            num_states,
                            num_states,
                            num_atoms,
                            3)

    return term_3


def decoherence_T_R(pot_V,
                    delta_R,
                    delta_P,
                    nacv,
                    mass,
                    vel,
                    hbar=1):

    # pot_V has dimension num_samples x num_states x num_states

    term_1 = -1j / hbar * commute(pot_V, delta_R)

    # mass has dimension `num_atoms`
    # `delta_P` has dimension num_samples x num_states
    # x num_states x num_atoms x 3

    term_2 = delta_P / mass.reshape(1, 1, 1, -1, 1)

    term_3 = get_term_3(nacv=nacv,
                        delta=delta_R,
                        vel=vel)

    T_R = term_1 + term_2 + term_3

    return T_R


def decoherence_T_ii(T,
                     surfs):
    T_ii = np.take_along_axis(arr=T,
                              indices=surfs.reshape(-1, 1, 1, 1, 1),
                              axis=1
                              ).squeeze(1)

    T_ii = np.take_along_axis(arr=T_ii,
                              indices=surfs.reshape(-1, 1, 1, 1),
                              axis=1
                              ).squeeze(1)

    num_states = T.shape[1]
    num_samples = T.shape[0]
    delta = np.eye(num_states).reshape(
        1,
        num_states,
        num_states,
        1,
        1)

    T_ii_delta = T_ii.reshape(num_samples, 1, 1, -1, 3) * delta

    return T_ii_delta


def deriv_delta_R(pot_V,
                  delta_R,
                  delta_P,
                  nacv,
                  mass,
                  vel,
                  surfs,
                  hbar=1,
                  **kwargs):

    T_R = decoherence_T_R(pot_V=pot_V,
                          delta_R=delta_R,
                          delta_P=delta_P,
                          nacv=nacv,
                          mass=mass,
                          vel=vel,
                          hbar=hbar)

    T_ii_delta = decoherence_T_ii(T=T_R,
                                  surfs=surfs)

    deriv = T_R - T_ii_delta

    return deriv


def get_F_alpha(force_nacv,
                forces):

    num_samples = force_nacv.shape[0]
    num_states = force_nacv.shape[1]
    num_atoms = force_nacv.shape[3]

    diag_idx = np.arange(num_states)
    row_idx = np.arange(num_states) * num_states
    idx = diag_idx + row_idx

    F_alpha = np.zeros((num_samples,
                        num_states * num_states,
                        num_atoms,
                        3))
    # forces on diagonal
    np.put_along_axis(arr=F_alpha,
                      indices=idx.reshape(1, -1, 1, 1),
                      values=forces,
                      axis=1)

    F_alpha = F_alpha.reshape(num_samples,
                              num_states,
                              num_states,
                              num_atoms,
                              3)

    # - force nacv on off-diagonal (force nacv is the
    # positive gradient so it needs a negative in front)

    # Make sure force_nacv has zeros on the diagonal
    F_alpha -= force_nacv

    return F_alpha


def get_F_alpha_sh(forces,
                   surfs):

    num_samples = forces.shape[0]
    num_states = forces.shape[1]
    num_atoms = forces.shape[-2]

    F_sh = np.take_along_axis(arr=forces,
                              indices=surfs.reshape(-1, 1, 1, 1),
                              axis=1)
    F_sh = F_sh.reshape(num_samples,
                        1,
                        1,
                        num_atoms,
                        3)
    id_elec = np.eye(num_states, num_states).reshape(1,
                                                     num_states,
                                                     num_states,
                                                     1,
                                                     1)

    F_sh_id = F_sh * id_elec

    return F_sh_id


def get_delta_F(force_nacv,
                forces,
                surfs):

    F_alpha = get_F_alpha(force_nacv=force_nacv,
                          forces=forces)

    F_alpha_sh = get_F_alpha_sh(forces=forces,
                                surfs=surfs)
    delta_F = F_alpha - F_alpha_sh

    return delta_F


def decoherence_T_P(pot_V,
                    delta_P,
                    nacv,
                    force_nacv,
                    forces,
                    surfs,
                    vel,
                    sigma,
                    hbar=1):

    term_1 = -1j / hbar * commute(pot_V, delta_P)

    delta_F = get_delta_F(force_nacv=force_nacv,
                          forces=forces,
                          surfs=surfs)
    term_2 = 1 / 2 * (matmul(delta_F, sigma)
                      + matmul(sigma, delta_F))

    term_3 = get_term_3(nacv=nacv,
                        delta=delta_P,
                        vel=vel)

    T_P = term_1 + term_2 + term_3

    return T_P  # , term_1, term_2, term_3


def deriv_delta_P(pot_V,
                  delta_P,
                  nacv,
                  force_nacv,
                  forces,
                  surfs,
                  vel,
                  sigma,
                  hbar=1,
                  **kwargs):

    T_P = decoherence_T_P(pot_V=pot_V,
                          delta_P=delta_P,
                          nacv=nacv,
                          force_nacv=force_nacv,
                          forces=forces,
                          surfs=surfs,
                          vel=vel,
                          sigma=sigma,
                          hbar=hbar)

    T_ii_delta = decoherence_T_ii(T=T_P,
                                  surfs=surfs)

    deriv = T_P - T_ii_delta

    return deriv


def deriv_sigma(pot_V,
                delta_R,
                nacv,
                force_nacv,
                forces,
                surfs,
                vel,
                sigma,
                hbar=1,
                **kwargs):

    F_alpha = get_F_alpha(force_nacv=force_nacv,
                          forces=forces)

    term_1 = -1j / hbar * commute(pot_V, sigma)
    term_2 = 1j / hbar * commute(F_alpha,
                                 delta_R).sum((-1, -2))
    # `vel` has shape num_samples x num_atoms x 3
    # `nacv` has shape num_samples x num_states x
    # num_states x num_atoms x 3

    num_samples = nacv.shape[0]
    num_atoms = nacv.shape[-2]

    vel_reshape = vel.reshape(num_samples,
                              1,
                              1,
                              num_atoms,
                              3)
    term_3 = (-commute(vel_reshape * nacv, sigma)
              .sum((-1, -2)))

    deriv = term_1 + term_2 + term_3

    return deriv


def get_delta_partials(pot_V,
                       delta_P,
                       delta_R,
                       nacv,
                       force_nacv,
                       forces,
                       surfs,
                       vel,
                       sigma,
                       mass,
                       hbar=1):

    partial_P = partial(deriv_delta_P,
                        pot_V=pot_V,
                        nacv=nacv,
                        force_nacv=force_nacv,
                        forces=forces,
                        surfs=surfs,
                        vel=vel,
                        hbar=hbar)

    # missing: delta_R, delta_P

    partial_R = partial(deriv_delta_R,
                        pot_V=pot_V,
                        nacv=nacv,
                        mass=mass,
                        vel=vel,
                        surfs=surfs,
                        hbar=hbar)

    # missing: delta_R, sigma

    partial_sigma = partial(deriv_sigma,
                            pot_V=pot_V,
                            nacv=nacv,
                            force_nacv=force_nacv,
                            forces=forces,
                            surfs=surfs,
                            vel=vel,
                            hbar=hbar)

    return partial_P, partial_R, partial_sigma


def runge_delta(pot_V,
                delta_P,
                delta_R,
                nacv,
                force_nacv,
                forces,
                surfs,
                vel,
                sigma,
                mass,
                elec_dt,
                hbar=1):

    derivs = get_delta_partials(pot_V=pot_V,
                                delta_P=delta_P,
                                delta_R=delta_R,
                                nacv=nacv,
                                force_nacv=force_nacv,
                                forces=forces,
                                surfs=surfs,
                                vel=vel,
                                sigma=sigma,
                                mass=mass,
                                hbar=hbar)

    init_vals = [delta_P, delta_R, sigma]
    intermed_vals = copy.deepcopy(init_vals)
    final_vals = copy.deepcopy(init_vals)
    num_vals = len(init_vals)

    step_size = np.array([0.5, 0.5, 1])
    final_weight = np.array([1, 2, 2, 1]) / 6
    names = ["delta_P", "delta_R", "sigma"]

    for i in range(4):
        kwargs = {name: val for name, val in
                  zip(names, intermed_vals)}
        k_i = [deriv(**kwargs)
               for deriv in derivs]

        intermed_vals = []
        for n in range(num_vals):
            if isinstance(final_vals[n], np.ndarray):
                final_vals[n] = final_vals[n].astype('complex128')

            final_vals[n] += k_i[n] * elec_dt * final_weight[i]
            if i == 3:
                continue
            intermed_vals.append(init_vals[n] + (
                k_i[n] * elec_dt * step_size[i]))

    return final_vals


def add_decoherence(c,
                    surfs,
                    new_surfs,
                    delta_P,
                    delta_R,
                    nacv,
                    energy,
                    forces,
                    mass):
    """
    Landry, B.R. and Subotnik, J.E., 2012. How to recover Marcus theory with 
    fewest switches surface hopping: Add just a touch of decoherence. The 
    Journal of chemical physics, 137(22), p.22A513.
    """

    pass


def get_other_surfs(surfs,
                    num_states,
                    num_samples):
    all_surfs = (np.arange(num_states).reshape(-1, 1)
                 .repeat(num_samples, 1).transpose())
    other_idx = all_surfs != surfs.reshape(-1, 1)
    other_surfs = all_surfs[other_idx].reshape(num_samples, -1)

    return other_surfs


def truhlar_decoherence(c,
                        surfs,
                        energy,
                        vel,
                        dt,
                        mass,
                        hbar=1,
                        C=0.1,
                        **kwargs):
    """
    Originally attributed to Truhlar, cited from
    G. Granucci and M. Persico. "Critical appraisal of the
    fewest switches algorithm for surface hopping."" J. Chem. Phys.,
    126, 134 114 (2007).
    """

    num_samples = c.shape[0]
    num_states = c.shape[1]

    other_surfs = get_other_surfs(surfs=surfs,
                                  num_states=num_states,
                                  num_samples=num_samples)

    c_m = np.take_along_axis(c,
                             surfs.reshape(-1, 1),
                             axis=-1)

    E_m = np.take_along_axis(energy,
                             surfs.reshape(-1, 1),
                             axis=-1)

    c_k = np.take_along_axis(c,
                             other_surfs,
                             axis=-1)

    E_k = np.take_along_axis(energy,
                             other_surfs,
                             axis=-1)

    # vel has shape num_samples x num_atoms x 3
    E_kin = (1 / 2 * mass.reshape(1, -1, 1) * vel ** 2).sum((-1, -2))
    tau_km = hbar / abs(E_k - E_m) * (1 + C / E_kin.reshape(-1, 1))
    c_k_prime = c_k * np.exp(-dt / tau_km)

    num = 1 - (abs(c_k_prime) ** 2).sum(-1)
    # in case some c_k's are slightly over 1 due to
    # numerical error

    num[num < 0] = 0

    c_m_prime = c_m * (
        num.reshape(-1, 1)
        / abs(c_m) ** 2
    ) ** 0.5

    new_c = np.zeros_like(c)
    np.put_along_axis(new_c,
                      surfs.reshape(-1, 1),
                      c_m_prime,
                      axis=-1)

    np.put_along_axis(new_c,
                      other_surfs,
                      c_k_prime,
                      axis=-1)

    return new_c
