import numpy as np
import random
import copy

from nff.md.utils_ax import atoms_to_nxyz
from nff.md.tully.io import (get_results, get_dc_dt,
                             get_p_hop)


class NeuralTully:
    def __init__(self,
                 atoms_list,
                 cutoff,
                 cutoff_skin,
                 device,
                 batch_size,
                 num_states,
                 surf,
                 dt,
                 max_gap_hop):
        """
        `max_gap_hop` in a.u.
        """

        self.atoms_list = atoms_list
        self.num_atoms = [len(atoms) for atoms
                          in self.atoms_list]
        self.num_samples = len(atoms_list)

        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.device = device
        self.batch_size = batch_size
        self.num_states = num_states
        self.surf = surf
        self.dt = dt

        self.nbr_list = None
        self.props_list = [{"c": self.init_c()}]
        self.max_gap_hop = max_gap_hop

    def init_c(self):
        c = np.zeros((self.num_samples, self.num_states))
        c[:, self.surf] = 1
        return c

    @property
    def c(self):
        return self.props_list[-1]["c"]

    @c.setter
    def c(self, value):
        self.props_list[-1]["c"] = value

    @property
    def nxyz_list(self):
        lst = [atoms_to_nxyz(atoms) for atoms in
               self.atoms_list]

        return lst

    @property
    def old_U(self):
        if not self.props_list:
            return
        return self.props_list[-1]["U"]

    @property
    def vel(self):
        vels = np.stack([atoms.get_velocities()
                         for atoms in self.atoms_list])
        return vels

    @property
    def props(self):
        return self.props_list[-1]

    @props.setter
    def props(self, value):
        self.props_list.append(value)

    def update_props(self,
                     needs_nbrs):

        props = get_results(model=self.model,
                            nxyz_list=self.nxyz_list,
                            nbr_list=self.nbr_list,
                            num_atoms=self.num_atoms,
                            needs_nbrs=needs_nbrs,
                            cutoff=self.cutoff,
                            cutoff_skin=self.cutoff_skin,
                            device=self.device,
                            batch_size=self.batch_size,
                            old_U=self.old_U,
                            num_states=self.num_states,
                            surf=self.surf,
                            max_gap_hop=self.max_gap_hop)

        self.props = props

    def get_new_surf(self, p_hop):

        # To avoid biasing in the direction of one hop vs. another,
        # we randomly shuffle the order of self.hopping_probabilities
        # each time.

        idx = list(range(self.num_states))
        random.shuffle(idx)

        new_surf = copy.deepcopy(self.surf)

        for i in idx:
            if i == self.surf:
                continue

            p = p_hop[idx]
            rnd = np.random.rand()

            hop = (p > rnd)
            if hop:
                new_surf = i
                break

        return new_surf

    def rescale(self):
        pass

    def decoherence(self):
        pass

    def step(self,
             needs_nbrs):

        self.update_props(needs_nbrs)

        dc_dt, T = get_dc_dt(c=self.c,
                             vel=self.vel,
                             results=self.props,
                             num_states=self.num_states)

        # need to step c, the positions, and the velocities,
        # and use different time-steps for each

        c = self.step_c(dc_dt)

        p_hop = get_p_hop(c=c,
                          T=T,
                          dt=self.dt,
                          surf=self.surf)

        new_surf = self.get_new_surf(p_hop)
        if new_surf != self.surf:
            self.rescale()
            self.surf = new_surf

        self.decoherence()
