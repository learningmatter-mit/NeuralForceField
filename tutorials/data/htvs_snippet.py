"""
Script to interface with htvs and the database to generate bond lists for different species.
"""

import numpy as np
import itertools
import sys
import os
import pdb

# change to your htvs directory
htvs_dir = "/home/saxelrod/htvs"

import django
djangochem_dir = os.path.join(htvs_dir, "djangochem")
sys.path.append(djangochem_dir)
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
os.environ["DJANGOCHEMDIR"] = djangochem_dir
django.setup()

from analysis.reacted_geometry import getrootconformers, get_component_order, DISTANCETHRESHOLDICT
from pgmols.models import Calc, Geom, Species, Stoichiometry, Method, Cluster



def get_all_clusters(spec_query):
    """Get a list of smiles of all the species that contains the species provided 
    
    Args:
        spec (Species Queryset): Description
    
    Returns:
        list: Description
    """
    smiles_list = []
    for spec in spec_query:
        smiles_list.append(spec.smiles)

        smiles_list += list( Species.objects.filter(components=spec).values_list("smiles", flat=True) )
    return smiles_list

def get_nuclei_bond_and_bondlen(geom):
    """get list of bond, atomicnums, bondlength
    
    Args:
        geom (Geom): Description
    
    Returns:
        list: bondlist, atomic numbers, bond length
    """
    atomicnums, adjmat, dmat = geom.adjdistmat(threshold=DISTANCETHRESHOLDICT)
    mol_bond = np.array(adjmat.nonzero()).transpose()#.tolist()
    mol_nuclei = [atomicnums.tolist()]
    mol_bondlen = dmat[mol_bond[:, 0], mol_bond[:, 1]]
    
    return [mol_bond.tolist(), mol_nuclei, mol_bondlen.tolist()]

def check_graph_reference_existence(smileslist, methodname='molecular_mechanics_mmff94', projectname='lipoly'):
    """check if the clusters or molecules in the database has a converged reference structure
    
    Args:
        smileslist (list): list of smiles
        methodname (str, optional): method name
        projectname (str, optional): project name
    
    Returns:
        TYPE: smiles that have a reference graph, smiles that do not
    """
    graph_ref_not_exist_smiles = []
    graph_ref_exist_smiles = []
    for smiles in smileslist:
        rootsp = Species.objects.filter(smiles=smiles,group__name=projectname).first().components.all()
        
        if rootsp.count() == 0:
            graph_ref_exist_smiles.append(smiles)
        
        for sp in rootsp:
            converged_geom_count = Geom.objects.filter(species=sp, 
                                                         converged=True, 
                                                         method__name=methodname
                                                        ).count()
            if converged_geom_count == 0:
                graph_ref_not_exist_smiles.append(smiles)
            else:
                graph_ref_exist_smiles.append(smiles)
                
    return  graph_ref_exist_smiles, graph_ref_not_exist_smiles


def gen_geomsrepeats_in_cluster(clustergeom, parentgeoms_query):   
    '''
        takes in geoms query and generate dictionary of subspecies count 
    '''
    
    repeats_dictionary = {}
    newrootparentgeoms = []

    for parent in parentgeoms_query:
        count = Cluster.objects.get(superspecies=clustergeom.species,
                                subspecies=parent.species).subcount
        repeats_dictionary[parent.id] = count
        newrootparentgeoms.append(parent)
        
    return repeats_dictionary


def gen_cluster_superlist(repeats_dictionary):
    
    '''
        For molecular clusters, generate a "deterministic reference partitioned 
        list of nulei type and a correponding adjacency matrix. This is used for 
        generate reference system for assigning bond list when placing classical 
        priors during training 
    '''
    
    partitioned_nuclei = []
    super_nuclei = []
    super_bond = []
    super_bondlen = []

    for geomid in repeats_dictionary: 
        geom = Geom.objects.filter(id=geomid).first()
        mol_ref = get_nuclei_bond_and_bondlen(geom)
        for i in range(repeats_dictionary[geomid]):    
            bond_list = np.array(mol_ref[0])
            nuclei_list = np.array(mol_ref[1])
            bond_len = mol_ref[2]
            
            # add the geom bond list to the super bond and shift index
            if len(bond_len) != 0:
                super_bond += (bond_list + np.array([[len( super_nuclei )]])).tolist()
                super_bondlen += bond_len
            
            # add nuclei to the super nuclei list, this has to be done 
            # after the super_bond is updated
            super_nuclei += nuclei_list[0].tolist()
            partitioned_nuclei += nuclei_list.tolist()
            
    return partitioned_nuclei, super_bond, super_bondlen


def get_mol_ref(smileslist,
                groupname='lipoly',
                method_name='molecular_mechanics_mmff94'):
    '''
        Obtain adjacency matrix and reference node order from goems 
        
        To do: bond list should only be stored once
    '''

    
    # check first if all the smiles has a valid reference graph given the method
    assert type(smileslist) == list
    smileslist, nographlist = check_graph_reference_existence(smileslist, methodname=method_name, projectname=groupname)
    if nographlist != []:
        raise Exception("{} has no reference graph for method name {} ".format(''.join(nographlist), method_name))
    

    mol_ref = dict()
    species = Species.objects.filter(group__name=groupname, smiles__in=smileslist)
    method = Method.objects.filter(name=method_name).first()
    
    # bond list is stored twice as directed edge 
    for sp in species:
        if sp.smiles not in mol_ref:  
            geom = Geom.objects.filter(species=sp
                                      ).order_by("calcs__props__totalenergy").first()
            # getting reference geoms as query sets 
            # What method should we use as default parent geoms 
            
            if '.' in sp.smiles: # This is a molecular cluster 
                rootparentgeoms = getrootconformers(geom=geom, method=method)
                # Use a function to get species count as a set                
                parentrepeatsdictionary = gen_geomsrepeats_in_cluster(geom, rootparentgeoms)

                # Generate reference nuclei list and its correponding molecular graph bond list 
                cluster_nuclei, cluster_bond, cluster_bondlen = gen_cluster_superlist(parentrepeatsdictionary)

                # update mol_ref dictionary 
                mol_ref[sp.smiles] = [cluster_bond, cluster_nuclei, cluster_bondlen]
            else:              
                mol_ref[sp.smiles] = get_nuclei_bond_and_bondlen(geom)
        
    return mol_ref

