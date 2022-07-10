import sys
import os
import shutil
import tempfile
from urllib import request as request
import numpy as np

from nff.data import Dataset

def get_md17_dataset(molecule, cutoff=5.0):
    """Download a dataset from MD17 and prepare in NFF format.
    
    Args:
        molecule (str): One of aspirin, benzene, ethanol, malonaldehyde, naphthalene, 
            salicylic, toluene, uracil, paracetamol, azobenzene
        cutoff (float): cutoff (Angstrom) for neighbor list construction.
    
    Returns:
        dataset (Dataset): NFF dataset.

    """

    smiles_dict = {'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
                'benzene': 'C1=CC=CC=C1',
                'ethanol': 'CCO',
                'malonaldehyde': 'O=CCC=O',
                'naphthalene': 'C1=CC=C2C=CC=CC2=C1',
                'salicylic': 'O=C(O)C1=CC=CC=C1O',
                'toluene': 'CC1=CC=CC=C1',
                'uracil': 'O=C1C=CNC(=O)N1',
                'paracetamol': 'CC(=O)NC1=CC=C(O)C=C1',
                'azobenzene': 'C1=CC=C(N=NC2=CC=CC=C2)C=C1'}
    
    if molecule not in smiles_dict.keys():
        raise ValueError(
            'Incorrect value for molecule. Must be one of: ', list(smiles_dict.keys())
        )

    # make tmpdir to save npz file
    tmpdir = tempfile.mkdtemp("MD")
    rawpath = os.path.join(tmpdir, molecule)
    url = (
        "http://www.quantum-machine.org/gdml/data/npz/"
        + f'{molecule}_dft.npz'
    )

    request.urlretrieve(url, rawpath)
    data = np.load(rawpath)
    shutil.rmtree(tmpdir)

    # get nxyz, energy, and forces
    numbers = data['z']
    force_data = data['F']
    energy_data = data['E']
    xyz_data = data['R']
    num_frames = energy_data.shape[0]
    nxyz_data = np.dstack((np.array([numbers]*num_frames).reshape(num_frames, -1, 1), np.array(xyz_data)))
    smiles_data = [smiles_dict[molecule]] * num_frames

    # convert forces to energy gradients
    props = {
    'nxyz': nxyz_data.tolist(),
    'energy': energy_data.tolist(),
    'energy_grad': [(-x).tolist() for x in force_data],
    'smiles': smiles_data
}

    # MD17 energies are in [kcal/mol] and forces are in [kcal/mol/angstrom]
    dataset = Dataset(props.copy(), units='kcal/mol')

    # generate neighborlist
    dataset.generate_neighbor_list(cutoff=cutoff)

    return dataset