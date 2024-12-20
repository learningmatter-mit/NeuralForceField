import itertools as itertools
from itertools import repeat
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import ModuleDict

from nff.train import load_model
from nff.utils.cuda import batch_to
from nff.utils.scatter import compute_grad


class ColVar(torch.nn.Module):
    """collective variable class

    computes cv and its Cartesian gradient
    """

    implemented_cvs = [
        "distance",
        "angle",
        "dihedral",
        "coordination_number",
        "coordination",
        "minimal_distance",
        "projecting_centroidvec",
        "projecting_veconplane",
        "projecting_veconplanenormal",
        "projection_channelnormal",
        "Sp",
        "Sd",
        "adjecencey_matrix",
        "energy_gap",
    ]

    def __init__(self, info_dict: dict):
        """initialization of many class variables to avoid recurrent assignment
        with every forward call
        Args:
            info_dict (dict): dictionary that contains all the definitions of the CV,
                              the common key is name, which defines the CV function
                              all other keys are specific to each CV
        """
        super(ColVar, self).__init__()
        self.info_dict = info_dict

        if "name" not in info_dict:
            raise TypeError('CV definition is missing the key "name"!')

        if self.info_dict["name"] not in self.implemented_cvs:
            raise NotImplementedError(f"The CV {self.info_dict['name']} is not implemented!")

        if self.info_dict["name"] == "Sp" or self.info_dict["name"] == "Sd":
            self.Oacid = torch.tensor(self.info_dict["x"])
            self.Owater = torch.tensor(self.info_dict["y"])
            self.H = torch.tensor(self.info_dict["z"])
            self.Box = torch.tensor(self.info_dict.get("box", None))
            self.O = torch.cat((Oacid, Owater))
            self.do = self.info_dict["dcv1"]
            self.d = self.info_dict["dcv2"]
            self.ro = self.info_dict["acidhyd"]
            self.r1 = self.info_dict["waterhyd"]

        elif self.info_dict["name"] == "adjecencey_matrix":
            self.model = self.info_dict["model"]
            self.device = self.info_dict["device"]
            self.bond_length = self.info_dict["bond_length"]
            self.cell = self.info_dict.get("box", None)
            self.atom_numbers = torch.tensor(self.info_dict["atom_numbers"])
            self.target = self.info_dict["target"]
            self.model = self.model.to(self.device)
            self.model.eval()

        elif self.info_dict["name"] == "projecting_centroidvec":
            self.vector_inds = self.info_dict["vector"]
            self.mol_inds = torch.LongTensor(self.info_dict["indices"])
            self.reference_inds = self.info_dict["reference"]

        elif (
            self.info_dict["name"] == "projecting_veconplane" or self.info_dict["name"] == "projecting_veconplanenormal"
        ):
            self.mol_inds = torch.LongTensor(self.info_dict["mol_inds"])
            self.ring_inds = torch.LongTensor(self.info_dict["ring_inds"])

        elif self.info_dict["name"] == "projection_channelnormal":
            self.mol_inds = torch.LongTensor(self.info_dict["mol_inds"])
            self.g1_inds = torch.LongTensor(self.info_dict["g1_inds"])
            self.g2_inds = torch.LongTensor(self.info_dict["g2_inds"])

        elif self.info_dict["name"] == "energy_gap":
            self.device = self.info_dict["device"]
            path = self.info_dict["path"]
            model_type = self.info_dict["model_type"]
            self.model = load_model(path, model_type=model_type, device=self.device)
            self.model = self.model.to(self.device)
            self.model.eval()

    def _get_com(self, indices: Union[int, list]) -> torch.tensor:
        """get center of mass (com) of group of atoms
        Args:
            indices (Union[int, list]): atom index or list of atom indices
        Returns:
            com (torch.tensor): Center of Mass
        """
        masses = torch.from_numpy(self.atoms.get_masses())

        if hasattr(indices, "__len__"):
            # compute center of mass for group of atoms
            center = torch.matmul(self.xyz[indices].T, masses[indices])
            m_tot = masses[indices].sum()
            com = center / m_tot

        else:
            # only one atom
            atom = int(indices)
            com = self.xyz[atom]

        return com

    def distance(self, index_list: list[Union[int, list]]) -> torch.tensor:
        """distance between two mass centers in range(0, inf)
        Args:
                distance beteen atoms: [ind0, ind1]
                distance between mass centers: [[ind00, ind01, ...], [ind10, ind11, ...]]
        Returns:
            cv (torch.tensor): computed distance
        """
        if len(index_list) != 2:
            raise ValueError("CV ERROR: Invalid number of centers in definition of distance!")

        p1 = self._get_com(index_list[0])
        p2 = self._get_com(index_list[1])

        # get distance
        r12 = p2 - p1
        cv = torch.linalg.norm(r12)

        return cv

    def angle(self, index_list: list[Union[int, list]]) -> torch.tensor:
        """get angle between three mass centers in range(-pi,pi)
        Args:
            index_list
                angle between two atoms: [ind0, ind1, ind3]
                angle between centers of mass: [[ind00, ind01, ...], [ind10, ind11, ...], [ind20, ind21, ...]]
        Returns:
            cv (torch.tensor): computed angle
        """
        if len(index_list) != 3:
            raise ValueError("CV ERROR: Invalid number of centers in definition of angle!")

        p1 = self._get_com(index_list[0])
        p2 = self._get_com(index_list[1])
        p3 = self._get_com(index_list[2])

        # get angle
        q12 = p1 - p2
        q23 = p2 - p3

        q12_n = torch.linalg.norm(q12)
        q23_n = torch.linalg.norm(q23)

        q12_u = q12 / q12_n
        q23_u = q23 / q23_n

        cv = torch.arccos(torch.dot(-q12_u, q23_u))

        return cv

    def dihedral(self, index_list: list[Union[int, list]]) -> torch.tensor:
        """torsion angle between four mass centers in range(-pi,pi)
        Params:
            self.info_dict['index_list']
                dihedral between atoms: [ind0, ind1, ind2, ind3]
                dihedral between center of mass: [[ind00, ind01, ...],
                                                  [ind10, ind11, ...],
                                                  [ind20, ind21, ...],
                                                  [ind30, ind 31, ...]]
        Returns:
            cv (float): computed torsional angle
        """
        if len(index_list) != 4:
            raise ValueError("CV ERROR: Invalid number of centers in definition of dihedral!")

        p1 = self._get_com(index_list[0])
        p2 = self._get_com(index_list[1])
        p3 = self._get_com(index_list[2])
        p4 = self._get_com(index_list[3])

        # get dihedral
        q12 = p2 - p1
        q23 = p3 - p2
        q34 = p4 - p3

        q23_u = q23 / torch.linalg.norm(q23)

        n1 = -q12 - torch.dot(-q12, q23_u) * q23_u
        n2 = q34 - torch.dot(q34, q23_u) * q23_u

        cv = torch.atan2(torch.dot(torch.cross(q23_u, n1), n2), torch.dot(n1, n2))

        return cv

    def coordination_number(self, index_list: list[int], switch_distance: float) -> torch.tensor:
        """coordination number between two atoms in range(0, 1)
        Args:
                distance between atoms: [ind00, ind01]
                switch_distance: value at which the switching function is 0.5
        Returns:
            cv (torch.tensor): computed distance
        """
        if len(index_list) != 2:
            raise ValueError("CV ERROR: Invalid number of atom in definition of coordination_number!")

        scaled_distance = self.distance(index_list) / switch_distance

        cv = (1.0 - scaled_distance.pow(6)) / (1.0 - scaled_distance.pow(12))

        return cv

    def coordination(self, index_list: list[list[int]], switch_distance: float) -> torch.tensor:
        """sum of coordination numbers between two sets of atoms in range(0, 1)
        Args:
                distance between atoms: [[ind00, ind01, ...], [ind10, ind11, ...]]
                switch_distance: value at which the switching function is 0.5
        Returns:
            cv (torch.tensor): computed distance
        """
        if len(index_list) != 2:
            raise ValueError("CV ERROR: Invalid number of atom lists in definition of coordination_number!")

        cv = torch.tensor(0.0)

        for idx1, idx2 in itertools.product(index_list[0], index_list[1]):
            cv = cv + self.coordination_number([idx1, idx2], switch_distance)

        return cv

    def minimal_distance(self, index_list: list[list[int]]) -> torch.tensor:
        """minimal distance between two sets of atoms
        Args:
                distance between atoms: [[ind00, ind01, ...], [ind10, ind11, ...]]
        Returns:
            cv (torch.tensor): computed distance
        """
        if len(index_list) != 2:
            raise ValueError("CV ERROR: Invalid number of atom lists in definition of minimal_distance!")

        distances = torch.zeros(len(index_list[0]) * len(index_list[1]))

        for ii, (idx1, idx2) in enumerate(itertools.product(index_list[0], index_list[1])):
            distances[ii] = self.distance([idx1, idx2])

        return distances.min()

    def projecting_centroidvec(self):
        """
        Projection of a position vector onto a reference vector
        Atomic indices are used to determine the coordiantes of the vectors.
        Params
        ------
        vector: list of int
           List of the indices of atoms that define the vector on which the position vector is projected
        indices: list if int
           List of indices of the mol/fragment
        reference: list of int
           List of atomic indices that are used as reference for the position vector
        """
        vector_pos = self.xyz[self.vector_inds]
        vector = vector_pos[1] - vector_pos[0]
        vector = vector / torch.linalg.norm(vector)
        mol_pos = self.xyz[self.mol_inds]
        reference_pos = self.xyz[self.reference_inds]
        mol_centroid = mol_pos.mean(axis=0)  # mol center

        reference_centroid = reference_pos.mean(axis=0)  # centroid of the whole structure

        # position vector with respect to the structure centroid
        rel_mol_pos = mol_centroid - reference_centroid

        # projection
        cv = torch.dot(rel_mol_pos, vector)
        return cv

    def projecting_veconplane(self):
        """
        Projection of a position vector onto a the average plane
        of an arbitrary ring defined in the structure
        Atomic indices are used to determine the coordiantes of the vectors.
        Params
        ------
        mol_inds: list of int
           List of indices of the mol/fragment tracked by the CV
        ring_inds: list of int
           List of atomic indices of the ring for which the average plane is calculated.
        """
        mol_coors = self.xyz[self.mol_inds]
        ring_coors = self.xyz[self.ring_inds]

        mol_cm = mol_coors.mean(axis=0)  # mol center
        ring_cm = ring_coors.mean(axis=0)  # ring center
        # ring atoms to center
        ring_coors = ring_coors - ring_cm

        r1 = torch.zeros(3, device=ring_coors.device)
        N = len(ring_coors)  # number of atoms in the ring
        for i, rl0 in enumerate(ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1 / N

        r2 = torch.zeros(3, device=ring_coors.device)
        for i, rl0 in enumerate(ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2 / N

        plane_vec = torch.cross(r1, r2)
        plane_vec = plane_vec / torch.linalg.norm(plane_vec)
        pos_vec = mol_cm - ring_cm

        cv = torch.dot(pos_vec, plane_vec)
        return cv

    def projecting_veconplanenormal(self):
        """
        Projection of a position vector onto the average plane
        of an arbitrary ring defined in the structure
        Atomic indices are used to determine the coordiantes of the vectors.
        Params
        ------
        mol_inds: list of int
           List of indices of the mol/fragment tracked by the CV
        ring_inds: list of int
           List of atomic indices of the ring for which the average plane is calculated.
        """

        mol_coors = self.xyz[self.mol_inds]
        ring_coors = self.xyz[self.ring_inds]

        mol_cm = mol_coors.mean(axis=0)  # mol center
        #         mol_cm     = self._get_com(self.mol_inds)
        ring_cm = ring_coors.mean(axis=0)  # ring center
        # ring atoms to center, center of geometry!
        ring_coors = ring_coors - ring_cm

        r1 = torch.zeros(3, device=ring_coors.device)
        N = len(ring_coors)  # number of atoms in the ring
        for i, rl0 in enumerate(ring_coors):
            r1 = r1 + rl0 * np.sin(2 * np.pi * i / N)
        r1 = r1 / N

        r2 = torch.zeros(3, device=ring_coors.device)
        for i, rl0 in enumerate(ring_coors):
            r2 = r2 + rl0 * np.cos(2 * np.pi * i / N)
        r2 = r2 / N

        # normalize r1 and r2
        r1 = r1 / torch.linalg.norm(r1)
        r2 = r2 / torch.linalg.norm(r2)
        # project position vector on r1 and r2
        pos_vec = mol_cm - ring_cm
        proj1 = torch.dot(pos_vec, r1)
        proj2 = torch.dot(pos_vec, r2)
        cv = proj1 + proj2
        return torch.abs(cv)

    def projection_channelnormal(self):
        """
        Projection of a position vector onto the vector
        along a channel
        Atomic indices are used to determine the coordiantes of the vectors.
        Params
        ------
        mol_inds: list of int
           List of indices of the mol/fragment tracked by the CV
        g1_inds: list of int
           List of atomic indices denoting "start" of channel
        g2_inds: list of int
           List of atomic indices denoting "end" of channel
        """

        self.xyz[self.mol_inds]
        g1_coors = self.xyz[self.g1_inds]
        g2_coors = self.xyz[self.g2_inds]

        mol_cm = self._get_com(self.mol_inds)
        center_g1 = g1_coors.mean(axis=0)
        center_g2 = g2_coors.mean(axis=0)
        center = (center_g1 + center_g2) / 2

        normal_vec = (center_g2 - center_g1) / torch.linalg.norm(center_g2 - center_g1)
        rel_pos = mol_cm - center

        cv = torch.dot(rel_pos, normal_vec)
        return cv

    def adjacency_matrix_cv(self):
        """Docstring"""
        edges, atomslist, Natoms, adjacency_matrix = get_adjacency_matrix(
            self.xyz, self.atom_numbers, self.bond_length, cell=self.cell, device=self.device
        )

        pred = self.model(atomslist, edges, Natoms, adjacency_matrix)[0]
        rmsd = (pred - self.target).norm()
        cv = rmsd.to("cpu").view(-1, 1)

        return cv

    def deproton1(self):
        """Emanuele Grifoni, GiovanniMaria Piccini, and Michele Parrinello, PNAS (2019), 116 (10) 4054-40
        https://www.pnas.org/doi/10.1073/pnas.1819771116

        Sp describes the proton exchange between acid-base pairs
        """

        dis_mat = self.xyz[None, :, :] - self.xyz[:, None, :]

        if Box is not None:
            cell_dim = Box.to(dis_mat.device)
            shift = torch.round(torch.divide(dis_mat, cell_dim))
            offsets = -shift
            dis_mat = dis_mat + offsets * cell_dim

        dis_sq = torch.linalg.norm(dis_mat, dim=-1)
        dis = dis_sq[self.O, :][:, self.H]

        dis1 = dis_sq[self.Oacid, :][:, self.Owater]
        cvmatrix = torch.exp(-self.do * dis)
        cvmatrix = cvmatrix / cvmatrix.sum(0)
        cvmatrixw = cvmatrix[self.Oacid.shape[0] :].sum(-1) - self.r1
        cvmatrix = cvmatrix[: self.Oacid.shape[0]].sum(-1) - self.ro
        cv1 = 2 * cvmatrix.sum() + cvmatrixw.sum()

        return cv1

    def deproton2(self):
        """Emanuele Grifoni, GiovanniMaria Piccini, and Michele Parrinello, PNAS (2019), 116 (10) 4054-40
        https://www.pnas.org/doi/10.1073/pnas.1819771116

        Sd describes tge distance between acid-base pairs
        """

        dis_mat = self.xyz[None, :, :] - self.xyz[:, None, :]

        if Box is not None:
            cell_dim = Box.to(dis_mat.device)
            shift = torch.round(torch.divide(dis_mat, cell_dim))
            offsets = -shift
            dis_mat = dis_mat + offsets * cell_dim

        dis_sq = torch.linalg.norm(dis_mat, dim=-1)
        dis = dis_sq[self.O, :][:, self.H]
        dis1 = dis_sq[self.Oacid, :][:, self.Owater]
        cvmatrix = torch.exp(-self.do * dis)
        cvmatrix = cvmatrix / cvmatrix.sum(0)
        cvmatrixx = torch.exp(-self.d * dis)
        cvmatrixx = cvmatrixx / cvmatrixx.sum(0)
        cvmatrixw = cvmatrixx[self.Oacid.shape[0] :].sum(-1) - self.r1
        cvmatrix = cvmatrixx[: self.Oacid.shape[0]].sum(-1) - self.ro
        torch.cat((cvmatrix, cvmatrixw))
        cvmatrix2 = torch.matmul(cvmatrix.view(1, -1).t(), cvmatrixw.view(1, -1))
        cvmatrix2 = -cvmatrix2 * dis1
        cv2 = cvmatrix2.sum()

        return cv2

    def energy_gap(self, enkey1, enkey2):
        """get energy gap betweentwo adiabatic PES
        Args:
            enkey1 (str): key of one adiabatic PES
            enkey2 (str): key of the other PES

        Returns:
                cv (torch.tensor): computed energy gap
        """

        batch = batch_to(self.atoms.get_batch(), self.device)
        pred = self.model(batch, device=self.device)
        energy_1 = pred[enkey1]
        energy_2 = pred[enkey2]
        e_diff = energy_2 - energy_1

        cv = torch.abs(e_diff)
        cv_grad = pred[enkey2 + "_grad"] - pred[enkey1 + "_grad"]
        if e_diff < 0:
            cv_grad *= -1.0

        return cv, cv_grad

    def forward(self, atoms):
        """switch function to call the right CV-func"""

        self.xyz = torch.from_numpy(atoms.get_positions())
        self.xyz.requires_grad = True

        self.atoms = atoms

        if self.info_dict["name"] == "distance":
            cv = self.distance(self.info_dict["index_list"])
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "angle":
            cv = self.angle(self.info_dict["index_list"])
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "dihedral":
            cv = self.dihedral(self.info_dict["index_list"])
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "coordination_number":
            cv = self.coordination_number(self.info_dict["index_list"], self.info_dict["switching_dist"])
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "coordination":
            cv = self.coordination(self.info_dict["index_list"], self.info_dict["switching_dist"])
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "minimal_distance":
            cv = self.minimal_distance(self.info_dict["index_list"])
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "projecting_centroidvec":
            cv = self.projecting_centroidvec()
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "projecting_veconplane":
            cv = self.projecting_veconplane()
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "projecting_veconplanenormal":
            cv = self.projecting_veconplanenormal()
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "projection_channelnormal":
            cv = self.projection_channelnormal()
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "Sp":
            cv = self.deproton1()
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "Sd":
            cv = self.deproton2()
            cv_grad = compute_grad(inputs=self.xyz, output=cv)

        elif self.info_dict["name"] == "energy_gap":
            cv, cv_grad = self.energy_gap(self.info_dict["enkey_1"], self.info_dict["enkey_2"])

        return cv.detach().cpu().numpy(), cv_grad.detach().cpu().numpy()


# implement SMILES to graph function
def smiles2graph(smiles):
    """
    Transfrom smiles into a list nodes (atomic number)

    Args:
        smiles (str): SMILES strings

    return:
        z(np.array), A (np.array): list of atomic numbers, adjancency matrix
    """

    mol = Chem.MolFromSmiles(smiles)  # no hydrogen
    z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    A = np.stack(Chem.GetAdjacencyMatrix(mol))
    # np.fill_diagonal(A,1)
    return z, A


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, AtomicNum_list, Edge_list, Natom_list, Adjacency_matrix_list):
        """
        GraphDataset object

        Args:
            z_list (list of torch.LongTensor)
            a_list (list of torch.LongTensor)
            N_list (list of int)


        """
        self.AtomicNum_list = AtomicNum_list  # atomic number
        self.Edge_list = Edge_list  # edge list
        self.Natom_list = Natom_list  # Number of atoms
        self.Adjacency_matrix_list = Adjacency_matrix_list

    def __len__(self):
        return len(self.Natom_list)

    def __getitem__(self, idx):
        AtomicNum = torch.LongTensor(self.AtomicNum_list[idx])
        Edge = torch.LongTensor(self.Edge_list[idx])
        Natom = self.Natom_list[idx]
        Adjacency_matrix = self.Adjacency_matrix_list[idx]

        return AtomicNum, Edge, Natom, Adjacency_matrix


def collate_graphs(batch):
    """Batch multiple graphs into one batched graph

    Args:

        batch (tuple): tuples of AtomicNum, Edge, Natom obtained from GraphDataset.__getitem__()

    Return
        (tuple): Batched AtomicNum, Edge, Natom

    """

    AtomicNum_batch = []
    Edge_batch = []
    Natom_batch = []
    Adjacency_matrix_batch = []
    cumulative_atoms = np.cumsum([0] + [b[2] for b in batch])[:-1]

    for i in range(len(batch)):
        z, a, N, A = batch[i]
        index_shift = cumulative_atoms[i]
        a = a + index_shift
        AtomicNum_batch.append(z)
        Edge_batch.append(a)
        Natom_batch.append(N)
        Adjacency_matrix_batch.append(A)

    AtomicNum_batch = torch.cat(AtomicNum_batch)
    Edge_batch = torch.cat(Edge_batch, dim=1)
    Natom_batch = Natom_batch
    # Adjacency_matrix_batch=torch.block_diag(*Adjacency_matrix_batch)
    Adjacency_matrix_batch = torch.cat(Adjacency_matrix_batch, dim=0).view(-1, 1)

    return AtomicNum_batch, Edge_batch, Natom_batch, Adjacency_matrix_batch


def scatter_add(src, index, dim_size, dim=-1, fill_value=0):
    """
    Sums all values from the src tensor into out at the indices specified in the index
    tensor along a given axis dim.
    """

    index_size = list(repeat(1, src.dim()))
    index_size[dim] = src.size(dim)
    index = index.view(index_size).expand_as(src)

    dim = range(src.dim())[dim]
    out_size = list(src.size())
    out_size[dim] = dim_size

    out = src.new_full(out_size, fill_value)

    return out.scatter_add_(dim, index, src)


class GNN(torch.nn.Module):
    """
    A GNN model
    """

    def __init__(self, n_convs=3, n_embed=64):
        super(GNN, self).__init__()
        self.atom_embed = nn.Embedding(100, n_embed)
        # Declare MLPs in a ModuleList
        self.convolutions = nn.ModuleList(
            [
                ModuleDict(
                    {
                        "update_mlp": nn.Sequential(
                            nn.Linear(n_embed, n_embed), nn.ReLU(), nn.Linear(n_embed, n_embed)
                        ),
                        "message_mlp": nn.Sequential(
                            nn.Linear(n_embed, n_embed), nn.ReLU(), nn.Linear(n_embed, n_embed)
                        ),
                    }
                )
                for _ in range(n_convs)
            ]
        )
        # Declare readout layers
        # self.readout = nn.Sequential(nn.Linear(n_embed, n_embed), nn.ReLU(), nn.Linear(n_embed, 1))

    def forward(self, AtomicNum, Edge, Natom, adjacency_matrix):
        ################ Code #################

        # Parametrize embedding
        h = self.atom_embed(AtomicNum)  # eqn. 1
        for conv in self.convolutions:
            messagei2j = conv.message_mlp(h[Edge[0]] * h[Edge[1]])
            messagei2j = messagei2j * adjacency_matrix
            # +  scatter_add(src=messagei2j, index=Edge[0], dim=0, dim_size=len(AtomicNum))
            node_message = scatter_add(src=messagei2j, index=Edge[1], dim=0, dim_size=len(AtomicNum))
            h = h + conv.update_mlp(node_message)
            output = [split.sum(0) for split in torch.split(h, Natom)]

        ################ Code #################
        return output


def adjfunc(x, m, s):
    return 4 / ((torch.exp(s * (x - m)) + 1) * (torch.exp((-s) * (x - m)) + 1))


def gauss(x, m, s, a, b):
    # return torch.exp(-abs((x-m)/2*s)**p)
    G = (1 + (2 ** (a / b) - 1) * abs((x - m) / s) ** a) ** (-b / a)
    G[torch.where(x < m)] = 1
    return G


def get_adjacency_matrix(xyz, atom_numbers, bond_length, oxygeninvolved, cell=None, device="cpu"):
    list(set(atom_numbers))
    dis_mat = xyz[None, :, :] - xyz[:, None, :]
    if cell is not None:
        cell_dim = torch.tensor(np.diag(cell))
        shift = torch.round(torch.divide(dis_mat, cell_dim))
        offsets = -shift

        dis_mat = dis_mat + offsets * cell_dim
    dis_sq = dis_mat.norm(dim=-1)
    bondlen = torch.ones(dis_sq.shape)
    bondlen[torch.where(atom_numbers == 8)[0], torch.where(atom_numbers == 14)[0].view(-1, 1)] = bond_length["8-14"]
    bondlen[torch.where(atom_numbers == 14)[0], torch.where(atom_numbers == 8)[0].view(-1, 1)] = bond_length["14-8"]
    bondlen[torch.where(atom_numbers == 8)[0], torch.where(atom_numbers == 1)[0].view(-1, 1)] = bond_length["8-1"]
    bondlen[torch.where(atom_numbers == 1)[0], torch.where(atom_numbers == 8)[0].view(-1, 1)] = bond_length["1-8"]
    # adjacency=dis_sq-bondlen
    # adjacency=(dis_sq-0.0001)/bondlen
    # adjacency_matrix=torch.exp(-((torch.abs(adjacency)/d)**p))
    # adjacency_matrix=(1-adjacency**m)/(1-adjacency**n)
    # adjacency_matrix=adjacency_matrix-torch.eye(adjacency_matrix.shape[0])
    adjacency_matrix = gauss(dis_sq, bondlen, 0.5, 2, 2)
    adjacency_matrix = adjacency_matrix[
        torch.where(atom_numbers == 14)[0].view(-1, 1), torch.where(atom_numbers == 8)[0][oxygeninvolved]
    ]
    adjacency_matrix = torch.matmul(adjacency_matrix, adjacency_matrix.t())
    adjacency_matrix = adjacency_matrix.fill_diagonal_(0)
    edges = torch.stack([i for i in torch.where(adjacency_matrix >= 0)])
    adjacency_matrix = adjacency_matrix[
        torch.where(adjacency_matrix >= 0)[0], torch.where(adjacency_matrix >= 0)[1]
    ].view(-1, 1)
    atomslist = torch.tensor([14 for i in torch.where(atom_numbers == 14)[0]]).view(-1)
    # molecules,edge_list,atom_list=get_molecules(xyz=xyz.detach(),atom_numbers=atom_numbers,bond_length=bond_length,periodic=False)
    # adjacency_matrix_list=[]
    # print(compute_grad(xyz,dis_sq[0,1]))
    # for i,m in enumerate(molecules):
    #    n=torch.tensor(m)
    #    adjacency_matrix_list.append(adjacency_matrix[edge_list[i][0],edge_list[i][1]].view(-1,1))
    return (
        torch.LongTensor(edges).to(device),
        torch.LongTensor(atomslist).to(device),
        [(len(atomslist))],
        adjacency_matrix.float().to(device),
    )


def get_molecules(atom, bond_length, mode="bond", periodic=True):
    types = list(set(atom.numbers))
    xyz = atom.positions
    # A=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0]))
    dis_mat = xyz[None, :, :] - xyz[:, None, :]
    if periodic:
        cell_dim = np.diag(np.array(atom.get_cell()))
        shift = np.round(np.divide(dis_mat, cell_dim))
        offsets = -shift
        dis_mat = dis_mat + offsets * cell_dim
    dis_sq = torch.tensor(dis_mat).pow(2).sum(-1).numpy()
    dis_sq = dis_sq**0.5
    clusters = np.array([0 for i in range(xyz.shape[0])])
    for i in range(xyz.shape[0]):
        mm = max(clusters)
        ty = atom.numbers[i]
        oxy_neighbors = []
        if mode == "bond":
            for t in types:
                if bond_length.get("%s-%s" % (ty, t)) is not None:
                    oxy_neighbors.extend(
                        list(
                            np.where(atom.numbers == t)[0][
                                np.where(dis_sq[i, np.where(atom.numbers == t)[0]] <= bond_length["%s-%s" % (ty, t)])[0]
                            ]
                        )
                    )
        elif mode == "cutoff":
            oxy_neighbors.extend(list(np.where(dis_sq[i] <= 6)[0]))
        oxy_neighbors = np.array(oxy_neighbors)
        if len(oxy_neighbors) == 0:
            clusters[i] = mm + 1
            continue
        if (clusters[oxy_neighbors] == 0).all() and clusters[i] != 0:
            clusters[oxy_neighbors] = clusters[i]
        elif (clusters[oxy_neighbors] == 0).all() and clusters[i] == 0:
            clusters[oxy_neighbors] = mm + 1
            clusters[i] = mm + 1
        elif (clusters[oxy_neighbors] == 0).all() == False and clusters[i] == 0:
            clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            clusters[oxy_neighbors] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
        elif (clusters[oxy_neighbors] == 0).all() == False and clusters[i] != 0:
            tmp = clusters[oxy_neighbors][clusters[oxy_neighbors] != 0][
                clusters[oxy_neighbors][clusters[oxy_neighbors] != 0]
                != min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            ]
            clusters[i] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            clusters[oxy_neighbors] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])
            for tr in tmp:
                clusters[np.where(clusters == tr)[0]] = min(clusters[oxy_neighbors][clusters[oxy_neighbors] != 0])

    molecules = []
    for i in range(1, max(clusters) + 1):
        if np.size(np.where(clusters == i)[0]) == 0:
            continue
        molecules.append(np.where(clusters == i)[0])

    return molecules


def reconstruct_atoms(atomsobject, mol_idx, centre=None):
    sys_xyz = torch.Tensor(atomsobject.get_positions(wrap=True))
    box_len = torch.Tensor(atomsobject.get_cell_lengths_and_angles()[:3])

    print(box_len)
    for idx in mol_idx:
        mol_xyz = sys_xyz[idx]
        center = mol_xyz.shape[0] // 2
        if centre is not None:
            center = centre
        intra_dmat = (mol_xyz[None, :, ...] - mol_xyz[:, None, ...])[center]
        if np.count_nonzero(atomsobject.cell.T - np.diag(np.diagonal(atomsobject.cell.T))) != 0:
            M, N = intra_dmat.shape[0], intra_dmat.shape[1]
            f = torch.linalg.solve(torch.Tensor(atomsobject.cell.T), (intra_dmat.view(-1, 3).T)).T
            g = f - torch.floor(f + 0.5)
            intra_dmat = torch.matmul(g, torch.Tensor(atomsobject.cell))
            intra_dmat = intra_dmat.view(M, 3)
            offsets = -torch.floor(f + 0.5).view(M, 3)
            traj_unwrap = mol_xyz + torch.matmul(offsets, torch.Tensor(atomsobject.cell))
        else:
            (intra_dmat > 0.5 * box_len).to(torch.float) * box_len
            add = (intra_dmat <= -0.5 * box_len).to(torch.float) * box_len
            shift = torch.round(torch.divide(intra_dmat, box_len))
            offsets = -shift
            traj_unwrap = mol_xyz + offsets * box_len
        # traj_unwrap=mol_xyz+add-sub
        sys_xyz[idx] = traj_unwrap

    new_pos = sys_xyz.numpy()

    return new_pos
