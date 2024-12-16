"""
Functions for performing a Tully time step
"""

import copy
from functools import partial

import numpy as np
import torch


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


def adiabatic_c(c,
                elec_substeps,
                old_H_plus_nacv,
                new_H_plus_nacv,
                dt,
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
            -1j * (
                old_H_plus_nacv + i / n *
                (new_H_plus_nacv - old_H_plus_nacv)
            )
            * delta_tau
        ).matrix_exp().numpy()
        exp = np.einsum('ijk, ikl -> ijl', exp, new_exp)

    P = exp
    c_new = np.einsum('ijk, ik -> ij', P, c)

    return c_new, P


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


def get_p_hop(hop_eqn='sharc',
              **kwargs):

    if hop_eqn == 'sharc':
        p = get_sharc_p(**kwargs)
    else:
        raise NotImplementedError

    return p


def get_sharc_p(old_c,
                new_c,
                P,
                surfs,
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
    denom = np.power(np.abs(c_beta_t), 2) - np.real(c_beta_dt * np.conj(P_beta_beta)
                                                    * np.conj(c_beta_t))
    pref = 1. - np.power(np.abs(c_beta_dt), 2) / (np.power(np.abs(c_beta_t), 2) + 1.e-8)

    h = np.zeros((num_samples, num_states))
    np.put_along_axis(h,
                      other_surfs,
                      pref * num / (denom + 1.e-8),
                      axis=-1)
    h[h < 0] = 0

    return h


def get_other_surfs(surfs,
                    num_states,
                    num_samples):
    all_surfs = (np.arange(num_states).reshape(-1, 1)
                 .repeat(num_samples, 1).transpose())
    other_idx = all_surfs != surfs.reshape(-1, 1)
    other_surfs = all_surfs[other_idx].reshape(num_samples, -1)

    return other_surfs


def try_hop(p_hop,
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


def truhlar_decoherence(c,
                        surfs,
                        energy,
                        vel,
                        dt,
                        mass,
                        hbar=1,
                        C=0.1,
                        eps=1.e-12,
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
    E_kin = (1 / 2 * mass.reshape(1, -1, 1) * np.power(vel, 2)).sum((-1, -2))
    # introduced espilon to keep abs(E_k - E_m) > 0
    # needed as energies for systems with SOCs=0 can be highly degenerate
    tau_km = hbar / abs(E_k - E_m + eps) * (1 + C / E_kin.reshape(-1, 1))
    c_k_prime = c_k * np.exp(-dt / tau_km)

    num = 1 - np.power(np.abs(c_k_prime), 2).sum(-1)
    # in case some c_k's are slightly over 1 due to
    # numerical error

    num[num < 0] = 0

    c_m_prime = c_m * np.sqrt(
        num.reshape(-1, 1)
        / np.power(np.abs(c_m), 2)
    )

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
