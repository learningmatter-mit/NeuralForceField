import torch
import numpy as np
import pdb
from scipy.spatial.distance import cdist

DISTANCETHRESHOLDICT = {
    (1., 1.): 1.00,
    (1., 3.): 1.30,
    (1., 5.): 1.50,
    (1., 6.): 1.30,
    (1., 7.): 1.30,
    (1., 8.): 1.30,
    (1., 9.): 1.30,
    (1., 11.): 1.65,
    (1., 14.): 1.65,
    (1., 16.): 1.50,
    (1., 17.): 1.60,
    (1., 35.): 1.60,
    (3., 6.): 1.60,
    (3., 7.): 1.60,
    (3., 8.): 1.60,
    (3., 9.): 1.60,
    (5., 11.): 1.8,
    (5., 17.): 2.1,
    (5., 35.): 2.1,
    (5., 6.): 1.70,
    (5., 7.): 1.70,
    (5., 8.): 1.70,
    (5., 9.): 1.70,
    (5., 11.): 1.70,
    (6., 6.): 1.70,
    (6., 8.): 1.62,
    (6., 9.): 1.65,
    (6., 11.): 1.70,
    (6., 14.): 2.10,
    (6., 16.): 2.20,
    (7., 8.): 1.55,
    (7., 11.): 1.70,
    (8., 11.): 1.70,
    (8., 16.): 2.10,
    (8., 17.): 1.80,
    (8., 8.): 1.70,
    (8., 9.): 1.50,
    (8., 14.): 1.85,
    (8., 35.): 1.70, }

DISTANCETHRESHOLD = 2.0

def get_neighbor_list(xyz, cutoff=5):
    """Get neighbor list from xyz positions of atoms.

    Args:
        xyz (torch.Tensor or np.array): (N, 3) array with positions
            of the atoms.
        cutoff (float): maximum distance to consider atoms as
            connected.

    Returns:
        nbr_list (torch.Tensor): (num_edges, 2) array with the
            indices of connected atoms.
    """

    # pdb.set_trace()

    xyz = torch.Tensor(xyz)
    n = xyz.size(0)

    # calculating distances
    dist = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)).pow(2).sum(dim=2).sqrt()

    # neighbor list
    mask = (dist <= cutoff)
    mask[np.diag_indices(n)] = 0
    nbr_list = mask.nonzero()

    return nbr_list


def distancemat(nxyz):
    xyz = np.array(nxyz)[:, 1:]
    atomicnums = np.array(nxyz)[:, 0].astype(int)
    return atomicnums, cdist(xyz, xyz)

def adjdistmat(nxyz, threshold=DISTANCETHRESHOLD):
    # pdb.set_trace()

    atomicnums, dmat = distancemat(nxyz)
    thresholdmat = np.array([[threshold.get(tuple(
        sorted((i, j))), DISTANCETHRESHOLD) for i in atomicnums] for j in atomicnums])
    adjmat = (dmat < thresholdmat).astype(int)
    np.fill_diagonal(adjmat, 0)
    return np.array(atomicnums), np.array(adjmat), np.array(dmat)

def get_bond_list(nxyz):

    atomicnums, adjmat, dmat = adjdistmat(nxyz=nxyz, threshold=DISTANCETHRESHOLDICT)
    mol_bond = np.array(adjmat.nonzero()).transpose()
    
    return mol_bond.tolist()

def list2adj(bond_list, size=None):
    # pdb.set_trace()

    E = bond_list
    if size is None:
        size = max(set([n for e in E for n in e])) + 1
    adjacency = [[0]*size for _ in range(size)]
    for sink, source in E:
        adjacency[sink][source] = 1
    return(adjacency)

def get_adj(nxyz, num_atoms):

    # pdb.set_trace()

    bond_list = get_bond_list(nxyz.tolist())
    adj = list2adj(bond_list, size=num_atoms)
    return adj

def get_bonded_neighbors(nxyz, num_atoms):

    A = torch.tensor(get_adj(nxyz, num_atoms))
    neighbors = A.nonzero()
    return neighbors


# geom_data.A = list2adj(ref_bond_list, size=num_atoms)












#     def list2adj(bond_list, size=None):
#         E = bond_list
#         if size is None:
#             size = max(set([n for e in E for n in e])) + 1
#         adjacency = [[0]*size for _ in range(size)]
#         for sink, source in E:
#             adjacency[sink][source] = 1
#         return(adjacency)

#     def get_component_order(nuclei, allparentnuclei):
#         possible_indices = list(itertools.permutations(
#             list(range(len(allparentnuclei)))))
#         if possible_indices == [(0,)]:
#             return((0, ), [[0, len(nuclei)]])
#         for indices in possible_indices:
#             jointnuclei = []
#             for j in indices:
#                 for i in allparentnuclei[j]:
#                     jointnuclei.append(i)
#             if len(jointnuclei) == len(nuclei):
#                 if (np.array(jointnuclei) == nuclei).all():
#                     finalindices = indices
#                     lens = [len(allparentnuclei[i]) for i in finalindices]
#                     lookingood = True
#                     startend = [[0, 0]]
#                     for i, length in enumerate(lens):
#                         startend[-1][-1] += length
#                         if i < len(lens) - 1:
#                             startend.append([startend[-1][-1], startend[-1][-1]])
#                     if lookingood:
#                         return(finalindices, startend)
#     def reorder(arr, index, n): 
#         temp = [0]*n; 
#         for i in range(0, n): 
#             temp[index[i]] = arr[i] 
#         for i in range(0, n): 
#             arr[i] = temp[i] 
#             index[i] = i 
#         return(arr)
#     # geom_data = Munch({'A': None, 'xyz': None, 'charges': None, 'E': None, 'E_ref': None, 'F': None})
#     # geom = Geom.objects.get(id=geom_id)

#     mol_ref = get_mol_ref(groupname=job_details.group, smileslist=get_smiles_list(geom_ids))
# from htvsneural.djangochem.analysis.reference_molecule_graph import get_mol_ref



# def get_mol_ref(smileslist, 
#                 groupname='lipoly',
#                 method_name='molecular_mechanics_mmff94'):
#     '''
#         Obtain adjacency matrix and reference node order from goems 
        
#         To do: bond list should only be stored once
#     '''
    
#     # check first if all the smiles has a valid reference graph given the method
#     assert type(smileslist) == list
#     smileslist, nographlist = check_graph_reference_existence(smileslist, methodname=method_name, projectname=groupname)
#     if nographlist != []:
#         raise Exception("{} has no reference graph for method name {} ".format(''.join(nographlist), method_name))
    

#     mol_ref = dict()
#     species = Species.objects.filter(group__name=groupname, smiles__in=smileslist)
#     method = Method.objects.filter(name=method_name).first()
    
#     # bond list is stored twice as directed edge 
#     for sp in species:
#         if sp.smiles not in mol_ref:  
#             geom = Geom.objects.filter(species=sp
#                                       ).order_by("calcs__props__totalenergy").first()
#             # getting reference geoms as query sets 
#             # What method should we use as default parent geoms 
            
#             if '.' in sp.smiles: # This is a molecular cluster 
#                 rootparentgeoms = getrootconformers(geom=geom, method=method)
#                 # Use a function to get species count as a set                
#                 parentrepeatsdictionary = gen_geomsrepeats_in_cluster(geom, rootparentgeoms)

#                 # Generate reference nuclei list and its correponding molecular graph bond list 
#                 cluster_nuclei, cluster_bond, cluster_bondlen = gen_cluster_superlist(parentrepeatsdictionary)

#                 # update mol_ref dictionary 
#                 mol_ref[sp.smiles] = [cluster_bond, cluster_nuclei, cluster_bondlen]
#             else:              
#                 mol_ref[sp.smiles] = get_nuclei_bond_and_bondlen(geom)
        
#     return mol_ref

# def check_graph_reference_existence(smileslist, methodname='molecular_mechanics_mmff94', projectname='lipoly'):
#     """check if the clusters or molecules in the database has a converged reference structure
    
#     Args:
#         smileslist (list): list of smiles
#         methodname (str, optional): method name
#         projectname (str, optional): project name
    
#     Returns:
#         TYPE: smiles that have a reference graph, smiles that do not
#     """
#     graph_ref_not_exist_smiles = []
#     graph_ref_exist_smiles = []
#     for smiles in smileslist:
#         rootsp = Species.objects.filter(smiles=smiles,group__name=projectname).first().components.all()
        
#         if rootsp.count() == 0:
#             graph_ref_exist_smiles.append(smiles)
        
#         for sp in rootsp:
#             converged_geom_count = Geom.objects.filter(species=sp, 
#                                                          converged=True, 
#                                                          method__name=methodname
#                                                         ).count()
#             if converged_geom_count == 0:
#                 graph_ref_not_exist_smiles.append(smiles)
#             else:
#                 graph_ref_exist_smiles.append(smiles)
                
#     return  graph_ref_exist_smiles, graph_ref_not_exist_smiles



# def get_nuclei_bond_and_bondlen(geom):
#     """get list of bond, atomicnums, bondlength
    
#     Args:
#         geom (Geom): Description
    
#     Returns:
#         list: bondlist, atomic numbers, bond length
#     """
#     atomicnums, adjmat, dmat = geom.adjdistmat(threshold=DISTANCETHRESHOLDICT)
#     mol_bond = np.array(adjmat.nonzero()).transpose()#.tolist()
#     mol_nuclei = [atomicnums.tolist()]
#     mol_bondlen = dmat[mol_bond[:, 0], mol_bond[:, 1]]
    
#     return [mol_bond.tolist(), mol_nuclei, mol_bondlen.tolist()]



# DISTANCETHRESHOLDICT = {
#     (1., 1.): 1.00,
#     (1., 3.): 1.30,
#     (1., 5.): 1.50,
#     (1., 6.): 1.30,
#     (1., 7.): 1.30,
#     (1., 8.): 1.30,
#     (1., 9.): 1.30,
#     (1., 11.): 1.65,
#     (1., 14.): 1.65,
#     (1., 16.): 1.50,
#     (1., 17.): 1.60,
#     (1., 35.): 1.60,
#     (3., 6.): 1.60,
#     (3., 7.): 1.60,
#     (3., 8.): 1.60,
#     (3., 9.): 1.60,
#     (5., 11.): 1.8,
#     (5., 17.): 2.1,
#     (5., 35.): 2.1,
#     (5., 6.): 1.70,
#     (5., 7.): 1.70,
#     (5., 8.): 1.70,
#     (5., 9.): 1.70,
#     (5., 11.): 1.70,
#     (6., 6.): 1.70,
#     (6., 8.): 1.62,
#     (6., 9.): 1.65,
#     (6., 11.): 1.70,
#     (6., 14.): 2.10,
#     (6., 16.): 2.20,
#     (7., 8.): 1.55,
#     (7., 11.): 1.70,
#     (8., 11.): 1.70,
#     (8., 16.): 2.10,
#     (8., 17.): 1.80,
#     (8., 8.): 1.70,
#     (8., 9.): 1.50,
#     (8., 14.): 1.85,
#     (8., 35.): 1.70, }


# # from pgmols:
# def adjdistmat(self, threshold=DISTANCETHRESHOLD):
#     atomicnums, dmat = self.distancemat()
#     if type(threshold) is dict:
#         thresholdmat = np.array([[threshold.get(tuple(
#             sorted((i, j))), DISTANCETHRESHOLD) for i in atomicnums] for j in atomicnums])
#         adjmat = (dmat < thresholdmat).astype(int)
#     else:
#         adjmat = (dmat < threshold).astype(int)
#     np.fill_diagonal(adjmat, 0)
#     return np.array(atomicnums), np.array(adjmat), np.array(dmat)
        


#     mol_ref = mol_ref[geom.species.smiles]
#     ref_bond_list, ref_node_order, ref_bond_len = mol_ref
#     num_atoms = len(list(itertools.chain.from_iterable(ref_node_order)))
#     finalindices, startend = get_component_order(np.array(geom.xyz)[:,0], mol_ref[1])
#     geom_data.A = list2adj(ref_bond_list, size=num_atoms)
#     geom_data.xyz = np.array(geom.xyz

