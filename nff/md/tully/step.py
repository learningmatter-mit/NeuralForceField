"""
Functions for performing a Tully time step
"""

import copy
import random
from functools import partial

import numpy as np
from nff.md.tully.io import get_p_hop, get_dc_dt

# TO-DO:
# - Add decoherence
# - Fix the hopping criterion for multiple states
# - Use Eq. (39) for the velocity re-scaling


def get_new_surf(p_hop,
                 num_states,
                 surfs):

    new_surfs = []

    for p, surf in zip(p_hop, surfs):

        # To avoid biasing in the direction of one hop vs. another,
        # we randomly shuffle the order of self.hopping_probabilities
        # each time.

        idx = list(range(num_states))
        random.shuffle(idx)

        new_surf = copy.deepcopy(surf)

        for i in idx:
            if i == surf:
                continue

            p = p_hop[idx]
            rnd = np.random.rand()

            hop = (p > rnd)
            if hop:
                new_surf = i
                break

        new_surfs.append(new_surf)
    new_surfs = np.array(new_surfs)

    return new_surfs


def rescale(energy,
            vel,
            nacv,
            mass,
            surfs,
            new_surfs):

    old_en = np.take_along_axis(energy, surfs.reshape(-1, 1),
                                -1).reshape(-1)
    new_en = np.take_along_axis(energy, new_surfs.reshape(-1, 1),
                                -1).reshape(-1)

    # `vel` has dimension num_samples x num_atoms x 3
    # `nacv` has dimension num_samples x num_states
    # x num_states x num_atoms x 3

    ones = [1] * 4
    start_nacv = np.take_along_axis(nacv, surfs
                                    .reshape(-1, *ones),
                                    axis=1)
    pair_nacv = np.take_along_axis(start_nacv, new_surfs
                                   .reshape(-1, *ones),
                                   axis=2
                                   ).squeeze(1).squeeze(1)

    norm = np.linalg.norm(pair_nacv, axis=-1)
    # for anything that didn't hop
    norm[norm == 0] = 1

    nac_dir = pair_nacv / norm.reshape(*pair_nacv.shape[:-1], 1)
    num_samples = vel.shape[0]

    v_par = ((nac_dir * vel).sum(-1)
             .reshape(num_samples, -1, 1) * nac_dir)

    # `mass` has dimension num_atoms
    # 1/2 mv^2 has dimension
    # num_samples x num_atoms x 3
    # `ke_par` has dimension num_samples

    ke_par = (1 / 2 * mass.reshape(1, -1, 1)
              * v_par ** 2).sum([-1, -2])

    # The scaling factor for the velocities
    # Has dimension `num_samples`

    scale = (old_en + ke_par - new_en) / ke_par

    # Anything less than zero leads to no hop
    scale[scale < 0] = np.nan
    scale = scale ** 0.5
    new_vel = scale * v_par + vel - v_par

    return new_vel


def try_hop(c,
            T,
            dt,
            surfs,
            vel,
            nacv,
            mass,
            energy):
    """
    `energy` has dimension num_samples x num_states
    """

    p_hop = get_p_hop(c=c,
                      T=T,
                      dt=dt,
                      surfs=surfs)

    num_states = energy.shape[-1]
    new_surfs = get_new_surf(p_hop=p_hop,
                             num_states=num_states,
                             surfs=surfs)

    new_vel = rescale(energy=energy,
                      vel=vel,
                      nacv=nacv,
                      mass=mass,
                      surfs=surfs,
                      new_surfs=new_surfs)

    # reset any frustrated hops
    frustrated = np.isnan(new_vel).any(-1).any(-1).nonzero()[0]
    new_vel[frustrated] = vel[frustrated]
    new_surfs[frustrated] = surfs[frustrated]

    return new_surfs, new_vel


def runge_c(c,
            vel,
            results,
            elec_dt,
            hbar=1):
    """
    Runge-Kutta step for c
    """

    deriv = partial(get_dc_dt,
                    vel=vel,
                    results=results,
                    hbar=hbar)

    k1, T1 = deriv(c)
    k2, T2 = deriv(c + elec_dt * k1 / 2)
    k3, T3 = deriv(c + elec_dt * k2 / 2)
    k4, T4 = deriv(c + elec_dt * k3)

    new_c = c + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return new_c, T1


def verlet_step_1(forces,
                  surfs,
                  vel,
                  xyz,
                  mass,
                  nuc_dt):

    # `forces` has dimension (num_samples x num_states
    # x num_atoms x 3)
    # `surfs` has dimension `num_samples`

    surf_forces = np.take_along_axis(
        forces, surfs.reshape(-1, 1, 1, 1),
        axis=1
    ).squeeze(1)

    # `surf_forces` has dimension (num_samples x
    #  num_atoms x 3)
    # `mass` has dimension `num_samples`
    accel = surf_forces / mass.reshape(-1, 1, 1)

    # `vel` and `xyz` each have dimension
    # (num_samples x num_atoms x 3)

    new_xyz = xyz + vel * nuc_dt + 0.5 * accel * nuc_dt ** 2
    new_vel = vel + 0.5 * nuc_dt * accel

    return new_xyz, new_vel


def verlet_step_2(forces,
                  surfs,
                  vel,
                  xyz,
                  mass,
                  nuc_dt):

    surf_forces = np.take_along_axis(
        forces, surfs.reshape(-1, 1, 1, 1),
        axis=1
    ).squeeze(1)

    accel = surf_forces / mass.reshape(-1, 1, 1)
    new_vel = vel + 0.5 * nuc_dt * accel

    return new_vel


def decoherence(c,
                surfs,
                new_surfs,
                delta_P,
                delta_R,
                nacv,
                energy,
                forces,
                mass):

    pass
