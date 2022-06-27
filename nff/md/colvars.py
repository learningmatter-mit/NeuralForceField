from typing import Union, Tuple
import numpy as np
import torch

from ase import Atoms
from nff.utils.scatter import compute_grad

""" Source for
- get_adjacency_matrix

"""

class ColVar(torch.nn.Module):
    """collective variable class
        
       computes cv and its Cartesian gradient
    """
    
    def __init__(self, info_dict: dict):
        """initialization of many class variables to avoid recurrent assignment
        with every forward call
        Args:
            info_dict (dict): dictionary that contains all the definitions of the CV
        """
        super(ColVar, self).__init__()
        self.info_dict=info_dict
        
        if 'name' not in info_dict.keys():
            raise TypeError("CV definition is missing the key \"name\"!")
        
        if self.info_dict['name'] == 'distance':
            self.idx_1 = self.info_dict['index_list'][0]
            self.idx_2 = self.info_dict['index_list'][1]
            
        elif self.info_dict['name'] == 'angle':
            self.idx_1 = self.info_dict['index_list'][0]
            self.idx_2 = self.info_dict['index_list'][1]
            self.idx_3 = self.info_dict['index_list'][2]
            
        elif self.info_dict['name'] == 'dihedral':
            self.idx_1 = self.info_dict['index_list'][0]
            self.idx_2 = self.info_dict['index_list'][1]
            self.idx_3 = self.info_dict['index_list'][2]
            self.idx_4 = self.info_dict['index_list'][3]
            
        elif self.info_dict['name'] == 'Sp':
            self.Oacid   = torch.tensor(self.info_dict['x'])
            self.Owater  = torch.tensor(self.info_dict['y'])
            self.H       = torch.tensor(self.info_dict['z'])
            self.Box     = torch.tensor(self.info_dict.get('box',None))
            self.O       = torch.cat((Oacid,Owater))
            self.do      = self.info_dict['dcv1']
            self.d       = self.info_dict['dcv2']
            self.ro      = self.info_dict['acidhyd']
            self.r1      = self.info_dict['waterhyd']
            
        elif self.info_dict['name'] == 'Sd':
            self.Oacid   = torch.tensor(self.info_dict['x'])
            self.Owater  = torch.tensor(self.info_dict['y'])
            self.H       = torch.tensor(self.info_dict['z'])
            self.Box     = torch.tensor(self.info_dict.get('box',None))
            self.O       = torch.cat((Oacid,Owater))
            self.do      = self.info_dict['dcv1']
            self.d       = self.info_dict['dcv2']
            self.ro      = self.info_dict['acidhyd']
            self.r1      = self.info_dict['waterhyd']
            
        elif self.info_dict['name'] == 'adjecencey_matrix':   
            self.model        = self.info_dict['model']
            self.device       = self.info_dict['device']
            self.bond_length  = self.info_dict['bond_length']
            self.cell         = self.info_dict.get('box',None)
            self.atom_numbers = torch.tensor(self.info_dict['atom_numbers'])
            self.target       = self.info_dict['target']
            self.model        = self.model.to(self.device)
            self.model.eval()
        else:
            raise NotImplementedError(f"The CV {self.info_dict['name']} is not implemented!")
        
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
            m_tot  = masses[indices].sum()
            com    = center / m_tot

        else:
            # only one atom
            atom = int(indices)
            com = self.xyz[atom]

        return com
    
    def distance(self) -> torch.tensor:
        """distance between two mass centers in range(0, inf)
        Args:
            self.info_dict['index_list']
                distance beteen atoms: [ind0, ind1]
                distance between mass centers: [[ind00, ind01, ...], [ind10, ind11, ...]]
        Returns:
            cv (torch.tensor): computed distance
        """
        if len(self.info_dict['index_list']) != 2:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of distance!"
            )

        p1 = self._get_com(self.idx_1)
        p2 = self._get_com(self.idx_2)

        # get distance
        r12 = p2 - p1
        cv = torch.linalg.norm(r12)

        return cv
    
    def angle(self) -> torch.tensor:
        """get angle between three mass centers in range(-pi,pi)
        Args:
            self.info_dict['index_list']
                angle between two atoms: [ind0, ind1, ind3]
                angle between centers of mass: [[ind00, ind01, ...], [ind10, ind11, ...], [ind20, ind21, ...]]
        Returns:
            cv (torch.tensor): computed angle
        """
        if len(self.info_dict['index_list']) != 3:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of angle!"
            )

        p1 = self._get_com(self.idx_1)
        p2 = self._get_com(self.idx_2)
        p3 = self._get_com(self.idx_3)

        # get angle
        q12 = p1 - p2
        q23 = p2 - p3

        q12_n = torch.linalg.norm(q12)
        q23_n = torch.linalg.norm(q23)

        q12_u = q12 / q12_n
        q23_u = q23 / q23_n

        cv = torch.arccos(torch.dot(-q12_u, q23_u))

        return cv
        
    def dihedral(self) -> torch.tensor:
        """torsion angle between four mass centers in range(-pi,pi)
        Args:
            self.info_dict['index_list']
                dihedral between atoms: [ind0, ind1, ind2, ind3]
                dihedral between center of mass: [[ind00, ind01, ...],
                                                  [ind10, ind11, ...],
                                                  [ind20, ind21, ...],
                                                  [ind30, ind 31, ...]]
        Returns:
            cv (float): computed torsional angle
        """
        if len(self.info_dict['index_list']) != 4:
            raise ValueError(
                "CV ERROR: Invalid number of centers in definition of dihedral!"
            )

        p1 = self._get_com(self.idx_1)
        p2 = self._get_com(self.idx_2)
        p3 = self._get_com(self.idx_3)
        p4 = self._get_com(self.idx_4)

        # get dihedral
        q12 = p2 - p1
        q23 = p3 - p2
        q34 = p4 - p3

        q23_u = q23 / torch.linalg.norm(q23)

        n1 = -q12 - torch.dot(-q12, q23_u) * q23_u
        n2 = q34 - torch.dot(q34, q23_u) * q23_u

        cv = torch.atan2(torch.dot(torch.cross(q23_u, n1), n2), torch.dot(n1, n2))
        
        return cv
        
    def adjacency_matrix_cv(self):
        """Docstring
        """
        edges, atomslist, Natoms, adjacency_matrix = get_adjacency_matrix(self.xyz, 
                                                                          self.atom_numbers, 
                                                                          self.bond_length, 
                                                                          cell=self.cell,
                                                                          device=self.device)
        
        pred   = self.model(atomslist, edges, Natoms, adjacency_matrix)[0]
        rmsd   = (pred-self.target).norm()
        cv     = rmsd.to('cpu').view(-1,1)
        
        return cv
    
    def deproton1(self, xyz):
        """Docstring
        """
        
        dis_mat = self.xyz[None, :, :] - self.xyz[:, None, :]
        
        if Box is not None:
            cell_dim = Box.to(dis_mat.device)
            shift    = torch.round(torch.divide(dis_mat,cell_dim))
            offsets  = -shift
            dis_mat  = dis_mat+offsets*cell_dim
            
        dis_sq = torch.linalg.norm(dis_mat, dim=-1)
        dis    = dis_sq[self.O,:][:,self.H]

        dis1      = dis_sq[self.Oacid,:][:,self.Owater]
        cvmatrix  = torch.exp(-self.do * dis)
        cvmatrix  = cvmatrix / cvmatrix.sum(0)
        cvmatrixw = cvmatrix[self.Oacid.shape[0]:].sum(-1) - self.r1
        cvmatrix  = cvmatrix[:self.Oacid.shape[0]].sum(-1) - self.ro
        cv1       = 2 * cvmatrix.sum() + cvmatrixw.sum()
        
        return cv2
    
    def deproton2(self):
        """Docstring
        """
        
        dis_mat = self.xyz[None, :, :] - self.xyz[:, None, :]
        
        if Box is not None:
            cell_dim = Box.to(dis_mat.device)
            shift    = torch.round(torch.divide(dis_mat,cell_dim))
            offsets  = -shift
            dis_mat  = dis_mat + offsets * cell_dim
            
        dis_sq    = torch.linalg.norm(dis_mat,dim=-1)
        dis       = dis_sq[self.O,:][:,self.H]
        dis1      = dis_sq[self.Oacid,:][:,self.Owater]
        cvmatrix  = torch.exp(-self.do * dis)
        cvmatrix  = cvmatrix / cvmatrix.sum(0)
        cvmatrixx = torch.exp(-self.d * dis)
        cvmatrixx = cvmatrixx / cvmatrixx.sum(0)
        cvmatrixw = cvmatrixx[self.Oacid.shape[0]:].sum(-1) - self.r1
        cvmatrix  = cvmatrixx[:self.Oacid.shape[0]].sum(-1) - self.ro
        cvmatrix1 = torch.cat((cvmatrix,cvmatrixw))
        cvmatrix2 = torch.matmul(cvmatrix.view(1,-1).t(),cvmatrixw.view(1,-1))
        cvmatrix2 = -cvmatrix2 * dis1
        cv2       = cvmatrix2.sum()
        
        return cv2
    
    def forward(self, atoms):
        """switch function to call the right CV-func
        """
        
        self.xyz = torch.from_numpy(atoms.get_positions())
        self.xyz.requires_grad=True
        
        self.atoms = atoms
        
        if self.info_dict['name'] == 'distance':
            cv = self.distance()
        elif self.info_dict['name'] == 'angle':
            cv = self.angle()
        elif self.info_dict['name'] == 'dihedral':
            cv = self.dihedral()
        elif self.info_dict['name'] == 'Sp':
            cv = self.deproton1()
        elif self.info_dict['name'] == 'Sd':
            cv = self.deproton2()
        
        cv_grad = compute_grad(inputs=self.xyz, output=cv)
        
        return cv.detach().numpy(), cv_grad.detach().numpy()