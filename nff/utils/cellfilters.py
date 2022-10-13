from ase.constraints import Filter, UnitCellFilter, ExpCellFilter
import numpy as np

from ase.stress import (full_3x3_to_voigt_6_stress,
                        voigt_6_to_full_3x3_stress)



class NeuralCellFilter(UnitCellFilter):

    # for geom optimization (only CellFilter that works for fix_cell=False)

    def update_nbr_list(self):
        return self.atoms.update_nbr_list()



class NeuralCellFilterDynamics(UnitCellFilter):

    # for dynamics (finite temperature)

    def update_nbr_list(self):
        return self.atoms.update_nbr_list()

    def get_temperature(self):
        return self.atoms.get_temperature()

    def get_kinetic_energy(self):
        return self.atoms.get_kinetic_energy()

    def get_moments_of_inertia(self, vectors=False):
        return self.atoms.get_moments_of_inertia(vectors=vectors)

    def get_angular_momentum(self):
        return self.atoms.get_angular_momentum()

    def get_center_of_mass(self):
        return self.atoms.get_center_of_mass()

    def get_velocities(self):
        return self.atoms.get_velocities()

    def set_velocities(self, velocities):
        return self.atoms.set_velocities(velocities=velocities)

    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        return self.atoms.set_cell(cell, scale_atoms=scale_atoms, apply_constraint=apply_constraint)

    def get_cell(self, complete=False):
        return self.atoms.get_cell(complete=complete)

    def set_positions(self, newpositions, apply_constraint=True):
        return self.atoms.set_positions(newpositions, apply_constraint=apply_constraint)

    def get_positions(self, wrap=False, **wrap_kw):
        return self.atoms.get_positions(wrap=wrap, **wrap_kw)

    def get_stress(self, voigt=True, apply_constraint=True, include_ideal_gas=False):

        stress = self.atoms.get_stress(voigt=voigt,
                                       apply_constraint=apply_constraint,
                                       include_ideal_gas=include_ideal_gas)

        volume = self.atoms.get_volume()
        virial = -volume * (voigt_6_to_full_3x3_stress(stress) +
                            np.diag([self.scalar_pressure] * 3))
        cur_deform_grad = self.deform_grad()
        virial = np.linalg.solve(cur_deform_grad, virial.T).T

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        stress = -full_3x3_to_voigt_6_stress(virial) / volume
        
        return stress

    def get_forces(self, **kwargs):
        """
        returns an array with shape (natoms,3) of the atomic forces.

        the first natoms rows are the forces on the atoms.
        """

        atoms_forces = self.atoms.get_forces(**kwargs)

        cur_deform_grad = self.deform_grad()
        forces = atoms_forces @ cur_deform_grad

        return forces

    def get_potential_energy(self, force_consistent=False, apply_constraint=True):
        return self.atoms.get_potential_energy(force_consistent=force_consistent, apply_constraint=apply_constraint)

    def get_global_number_of_atoms(self):
        return self.atoms.get_global_number_of_atoms()