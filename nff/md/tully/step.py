"""
Functions for performing a Tully time step
"""

import copy
import random

import numpy as np


def get_new_surf(p_hop,
                 num_states,
                 surf):

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

    return new_surf

def is_frustrated():
    pass

