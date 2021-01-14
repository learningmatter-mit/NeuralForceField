import requests
import numpy as np
import requests
import sys
import os 

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "../../../")

import nff.data as d
from sklearn.utils import shuffle


def get_MD17data(mol):

    data_url = {'ethanol_dft': 'http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz',
                'aspirin_dft': 'http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz',
                'benzene_dft': 'http://quantum-machine.org/gdml/data/npz/benzene_old_dft.npz',
                'uracil_dft': 'http://quantum-machine.org/gdml/data/npz/uracil_dft.npz',
                'naphthalene_dft': 'http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz',
                'salicylic_dft': 'http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz', 
                'malonaldehyde_dft': 'http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz',
                'toluene_dft': 'http://quantum-machine.org/gdml/data/npz/toluene_dft.npz'}
    
    fname = './{}.npz'.format(mol)
    
    if not os.path.isfile(fname):
        myfile = requests.get(data_url[mol])
        open(fname, 'wb').write(myfile.content)
    
    data = np.load(fname.format(mol))
    return data

def pack_MD17data(data, n_data=1000):
    
    size = data.f.R.shape[0]
    idx = shuffle(list(range(size)))[:n_data]
    
    nxyz_data = np.dstack((np.array([data.f.z] * n_data).reshape(n_data, -1, 1), np.array(data.f.R)[idx]))
    force_data = data.f.F[idx]
    #smiles_data = [data.f.name.tolist()] * n_data
    
    props = {'nxyz': nxyz_data.tolist(),
                'energy_grad': [(-x).tolist() for x in force_data],
                #'smiles': smiles_data
              }

    dataset = d.Dataset(props.copy(), units='kcal/mol')
    dataset.generate_neighbor_list(cutoff=5)
    
    return dataset