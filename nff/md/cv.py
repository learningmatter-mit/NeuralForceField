import torch
import numpy as np

class ProjVectorCentroid:
    """
    Collective variable class. Projection of a position vector onto a reference vector
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    vector: list of int
       List of the indices of atoms that define the vector on which the position vector is projected
    indices: list if int
       List of indices of the mol/fragment
    reference: list of int
       List of atomic indices that are used as reference for the position vector
       
    note: the position vector is calculated in the method get_value
    """
    def __init__(self, vector=[], indices=[], reference=[], device='cpu'):
        self.vector_inds = vector
        self.mol_inds = torch.LongTensor(indices)
        self.reference_inds = reference
    
    def get_value(self, positions):
        vector_pos = positions[self.vector_inds]
        vector = vector_pos[1] - vector_pos[0]
        vector = vector / torch.linalg.norm(vector)
        mol_pos = positions[self.mol_inds]
        reference_pos = positions[self.reference_inds]
        mol_centroid = mol_pos.mean(axis=0) # mol center
        reference_centroid = reference_pos.mean(axis=0) # centroid of the whole structure
        
        # position vector with respect to the structure centroid
        rel_mol_pos = mol_centroid - reference_centroid 
        
        # projection
        cv = torch.dot(rel_mol_pos, vector)
        return cv


class ProjVectorPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.
       
    note: the position vector is calculated in the method get_value
    """
    def __init__(self, mol_inds = [], ring_inds = []):
        self.mol_inds = torch.LongTensor(mol_inds) # list of indices
        self.ring_inds = torch.LongTensor(ring_inds) # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = [] 
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0) # mol center
        ring_cm = self.ring_coors.mean(axis=0) # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors) # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1/N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2/N

        plane_vec = torch.cross(r1, r2)
        plane_vec = plane_vec / torch.linalg.norm(plane_vec)
        pos_vec = mol_cm - ring_cm

        cv = torch.dot(pos_vec, plane_vec)
        return cv

class ProjOrthoVectorsPlane:
    """
    Collective variable class. Projection of a position vector onto a the average plane
    of an arbitrary ring defined in the structure
    Atomic indices are used to determine the coordiantes of the vectors.
    Params
    ------
    mol_inds: list of int
       List of indices of the mol/fragment tracked by the CV
    ring_inds: list of int
       List of atomic indices of the ring for which the average plane is calculated.
       
    note: the position vector is calculated in the method get_value
    """
    def __init__(self, mol_inds = [], ring_inds = []):
        self.mol_inds = torch.LongTensor(mol_inds) # list of indices
        self.ring_inds = torch.LongTensor(ring_inds) # list of indices
        # both self.mol_coors and self.ring_coors torch tensors with atomic coordinates
        # initiallized as list but will be set to torch tensors with set_positions
        self.mol_coors = [] 
        self.ring_coors = []

    def set_positions(self, positions):
        # update coordinate torch tensors from the positions tensor
        self.mol_coors = positions[self.mol_inds]
        self.ring_coors = positions[self.ring_inds]

    def get_indices(self):
        return self.mol_inds + self.ring_inds

    def get_value(self, positions):
        """Calculates the values of the CV for a specific atomic positions

        Args:
            positions (torch tensor): atomic positions

        Returns:
            float: current values of the collective variable
        """
        self.set_positions(positions)
        mol_cm = self.mol_coors.mean(axis=0) # mol center
        ring_cm = self.ring_coors.mean(axis=0) # ring center
        # ring atoms to center
        self.ring_coors = self.ring_coors - ring_cm

        r1 = torch.zeros(3, device=self.ring_coors.device)
        N = len(self.ring_coors) # number of atoms in the ring
        for i, rl0 in enumerate(self.ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1/N

        r2 = torch.zeros(3, device=self.ring_coors.device)
        for i, rl0 in enumerate(self.ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2/N

        # normalize r1 and r2
        r1 = r1 / torch.linalg.norm(r1)
        r2 = r2 / torch.linalg.norm(r2)
        # project position vector on r1 and r2
        pos_vec = mol_cm - ring_cm
        proj1 = torch.dot(pos_vec, r1)
        proj2 = torch.dot(pos_vec, r2)
        cv = proj1 + proj2
        return abs(cv)
