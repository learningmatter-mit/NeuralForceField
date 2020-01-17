import sys
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader
from ase import io
import numpy as np
import nff
import ase
from nff.io import AtomsBatch

DISTANCETHRESHOLDICT_SYMBOL = {
    ("H", "H"): 1.00,
    ("H", "Li"): 1.30,
    ("H", "N"): 1.50,
    ("H", "C"): 1.30,
    ("H", "N"): 1.30,
    ("H", "O"): 1.30,
    ("H", "F"): 1.30,
    ("H", "Na"): 1.65,
    ("H", "Si"): 1.65,
    ("H", "Mg"): 1.40,
    ("H", "S"): 1.50,
    ("H", "Cl"): 1.60,
    ("H", "Br"): 1.60,
    ("Li", "C"): 0.0,
    ("Li", "N"): 0.0,
    ("Li", "O"): 0.0,
    ("Li", "F"): 0.0,
    ("Li", "Mg"): 0.0,
    ("B", "C"): 1.70,
    ("B", "N"): 1.70,
    ("B", "O"): 1.70,
    ("B", "F"): 1.70,
    ("B", "Na"): 1.8,
    ("B", "Mg"): 1.8,
    ("B", "Cl"): 2.1,
    ("B", "Br"): 2.1,
    ("C", "C"): 1.70,
    ("C", "O"): 1.70,
    ("C", "N"): 1.8,
    ("C", "F"): 1.65,
    ("C", "Na"): 1.80,
    ("C", "Mg"): 1.70,
    ("C", "Si"): 2.10,
    ("C", "S"): 2.20,
    ("N", "O"): 1.55,
    ("N", "Na"): 1.70,
    ("N", "S"): 2.0,
    ("O", "Na"): 1.70,
    ("O", "Mg"): 1.35,
    ("O", "S"): 2.00,
    ("O", "Cl"): 1.80,
    ("O", "O"): 1.70,
    ("O", "F"): 1.50,
    ("O", "Si"): 1.85,
    ("O", "Br"): 1.70,
    ("F", "Mg"): 1.35, }

DISTANCETHRESHOLDICT_Z = {
    (1., 1.): 1.00,
    (1., 3.): 1.30,
    (1., 5.): 1.50,
    (1., 6.): 1.30,
    (1., 7.): 1.30,
    (1., 8.): 1.30,
    (1., 9.): 1.30,
    (1., 11.): 1.65,
    (1., 14.): 1.65,
    (1., 12.): 1.40,
    (1., 16.): 1.50,
    (1., 17.): 1.60,
    (1., 35.): 1.60,
    (3., 6.): 0.0,
    (3., 7.): 0.0,
    (3., 8.): 0.0,
    (3., 9.): 0.0,
    (3., 12.): 0.0,
    (5., 6.): 1.70,
    (5., 7.): 1.70,
    (5., 8.): 1.70,
    (5., 9.): 1.70,
    (5., 11.): 1.8,
    (5., 12.): 1.8,
    (5., 17.): 2.1,
    (5., 35.): 2.1,
    (6., 6.): 1.70,
    (6., 8.): 1.70,
    (6., 7.): 1.8,
    (6., 9.): 1.65,
    (6., 11.): 1.80,
    (6., 12.): 1.70,
    (6., 14.): 2.10,
    (6., 16.): 2.20,
    (7., 8.): 1.55,
    (7., 11.): 1.70,
    (7., 16.): 2.0,
    (8., 11.): 1.70,
    (8., 12.): 1.35,
    (8., 16.): 2.00,
    (8., 17.): 1.80,
    (8., 8.): 1.70,
    (8., 9.): 1.50,
    (8., 14.): 1.85,
    (8., 35.): 1.70,
    (9., 12.): 1.35 }


def get_dist_mat(xyz, box_len, unwrap=True):
    dis_mat = (xyz[:, None, :] - xyz[None, ...])

    # build minimum image convention 
    mask_pos = dis_mat.ge(0.5*box_len).type(torch.FloatTensor)
    mask_neg = dis_mat.lt(-0.5*box_len).type(torch.FloatTensor)

    # modify distance
    if unwrap:
        dis_add = mask_neg * box_len
        dis_sub = mask_pos * box_len
        dis_mat = dis_mat + dis_add - dis_sub

    # create cutoff mask
    dis_sq = dis_mat.pow(2).sum(-1)                  # compute squared distance of dim (B, N, N)
    #mask = (dis_sq <= cutoff ** 2) & (dis_sq != 0)                 # byte tensor of dim (B, N, N)
    #A = mask.unsqueeze(3).type(torch.FloatTensor).to(self.device) #         

    # 1) PBC 2) # gradient of zero distance 
    dis_sq = dis_sq.unsqueeze(-1)
    #dis_sq = (dis_sq * A) + 1e-8# to make sure the distance is not zero, otherwise there will be inf gradient 
    dis_mat = dis_sq.sqrt().squeeze()
    
    return dis_mat

def adjdistmat(atoms, threshold=DISTANCETHRESHOLDICT_Z, unwrap=True):
    #dmat = (xyz[:, None, :] - xyz[None, ...]).pow(2).sum(-1).numpy()
    xyz= torch.Tensor(atoms.get_positions(wrap=True))
    atomicnums = atoms.get_atomic_numbers().tolist()
    box_len = torch.Tensor( np.diag(atoms.get_cell()) )
    
    dmat = get_dist_mat(xyz, box_len, unwrap=unwrap).numpy()
    
    thresholdmat = np.array([[threshold.get(tuple(
        sorted((i, j))), 2.0) for i in atomicnums] for j in atomicnums])
    adjmat = (dmat < thresholdmat).astype(int)

    np.fill_diagonal(adjmat, 0)
    return np.array(atomicnums), np.array(adjmat), np.array(dmat), thresholdmat

def generate_mol_atoms(atomic_nums, xyz, cell):
    return Atoms(numbers=atomic_nums, positions=xyz, cell=cell, pbc=True)

def generate_subgraphs(atomsobject, unwrap=True, get_edge=False):
    
    atoms = AtomsBatch(atomsobject)
    z, adj, dmat,  threshold = adjdistmat(atoms, unwrap=unwrap)
    box_len = torch.Tensor( np.diag(atoms.get_cell()) )
    G=nx.from_numpy_matrix(adj)

    for i, item in enumerate(z):
        G.nodes[i]['z'] = item

    sub_graphs = nx.connected_component_subgraphs(G)
    
    edge_list = []
    partitions = []
    
    for i, sg in enumerate(sub_graphs):
        partitions.append(list(sg.nodes))
        if get_edge:
            edge_list.append(list(sg.edges))
        
    #print("found {} molecules".format(len(partitions)))
    
    if len(edge_list) != 0:
        return partitions, edge_list
    else:
        return partitions

def pop_molecules(atomsobject, mol_idx, id_list):
    xyz = atomsobject.get_positions()
    
    new_z = []
    new_pos = []
    new_partitions = []
    
    old_pos = atomsobject.get_positions()
    old_z = atomsobject.get_positions()
    
    for i, mol in enumerate( mol_idx ):
        if i not in id_list:
            new_z.append(atomsobject.get_atomic_numbers()[mol])
            new_pos.append(atomsobject.get_positions()[mol])
            new_partitions.append(mol)
        
    return Atoms(numbers = np.concatenate(new_z), positions= np.concatenate(new_pos), 
                 cell=atomsobject.cell, pbc=True)

def get_single_molecule(atomsobject, mol_idx, single_mol_id):    
    z = atomsobject.get_atomic_numbers()[mol_idx[single_mol_id]]
    pos = atomsobject.get_positions()[mol_idx[single_mol_id]]
    return Atoms(numbers = z, positions=pos, 
                 cell=atomsobject.cell, pbc=True)

def reconstruct_atoms(atomsobject, mol_idx):
    sys_xyz = torch.Tensor(atomsobject.get_positions(wrap=True) ) 
    box_len = torch.Tensor( atomsobject.get_cell_lengths_and_angles()[:3] )
    
    print(box_len)
    for idx in mol_idx:
        mol_xyz = sys_xyz[idx] 
        center = mol_xyz.shape[0]//2
        intra_dmat = (mol_xyz[None, ...]  - mol_xyz[:, None, ...])[center]
        sub = (intra_dmat > 0.5 * box_len).to(torch.float) * box_len
        add = (intra_dmat <= -0.5 * box_len).to(torch.float) * box_len
        traj_unwrap = mol_xyz + add - sub
        sys_xyz[idx] = traj_unwrap

    new_pos = sys_xyz.numpy()
    
    return new_pos


def list2adj(bond_list, size=None):
    E = bond_list
    if size is None:
        size = max(set([n for e in E for n in e])) + 1
    # make an empty adjacency list  
    adjacency = [[0]*size for _ in range(size)]
    # populate the list for each edge
    for sink, source in E:
        adjacency[sink][source] = 1
    return adjacency