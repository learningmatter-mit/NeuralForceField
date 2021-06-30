"""
Functions for performing a Tully time step
"""

import copy
import random

import numpy as np
from nff.md.tully.io import get_p_hop


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

    # `nacv` and `vel` each have dimension
    # num_samples x num_atoms x 3

    norm = np.linalg.norm(nacv, axis=-1)
    nac_dir = nacv / norm.reshape(*nacv.shape[:-1], 1)
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
