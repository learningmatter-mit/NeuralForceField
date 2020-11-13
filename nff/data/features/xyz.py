"""
Tools for generating xyz-baed features
"""

import logging

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import torch
from e3fp.pipeline import fprints_from_mol

from tqdm import tqdm


def get_conformer(xyz):
    """
    Create an RDKit conformer object from an xyz.
    Args:
        xyz (torch.Tensor): atom type and xyz of geometry.
    Returns:
        conformer (rdkit.Chem.Conformer): RDKit conformer
    """
    n_atoms = len(xyz)
    # initialize a conformer of the right length
    conformer = Chem.Conformer(n_atoms)
    # set each atom's positions in the order of the xyz
    for i, nxyz in zip(range(n_atoms), xyz):
        conformer.SetAtomPosition(i, nxyz[1:])
    return conformer


def get_mol(xyz, smiles, with_conformer=True):
    """
    Get an RDKit mol from an xyz and smiles. Note that this
    assumes that the xyz is ordered in the same way an RDKit
    object of the same smiles would be ordered, and that there
    is no change in connectivity between the RDKit mol and the
    xyz.

    Args:
        xyz (torch.Tensor): atom type and xyz of geometry.
        smiles (str): SMILES string
        with_conformer (bool): also add conformer to the RDKit mol
    Returns:
        mol (rdkit.Chem.rdchem.Mol): RDKit mol object
    """

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if with_conformer:
        conformer = get_conformer(xyz)
    mol.AddConformer(conformer)

    return mol


def get_3d_representation(xyz, smiles, method, mol=None):
    """
    Args:
        xyz (torch.Tensor): atom type and xyz of geometry.
        smiles (str): SMILES string
        method (str): RDKit method for 3D representation
        mol (rdkit.Chem.rdchem.Mol): RDKit mol object
    Returns:
        result (np.array): fingerprint
    """

    representation_fn = {
        'autocorrelation_3d': rdMD.CalcAUTOCORR3D,
        'rdf': rdMD.CalcRDF,
        'morse': rdMD.CalcMORSE,
        'whim': rdMD.CalcWHIM,
        'getaway': lambda x: rdMD.CalcWHIM(x, precision=0.001)
    }

    # if a `mol` is not given, generate it from the xyz and smiles
    if mol is None:
        mol = get_mol(xyz=xyz, smiles=smiles)
    fn = representation_fn[method]
    result = fn(mol)

    return result


def featurize_rdkit(dataset, method):
    """
    Featurize a dataset with RDKit 3D descriptors.
    Args:
        dataset (nff.data.dataset): NFF dataset
        method (str): RDKit method for 3D representation
    Returns:
        None
    """
    dataset.props[method] = []
    props = dataset.props

    # go through each geometry
    for i in range(len(props['nxyz'])):

        smiles = props['smiles'][i]
        nxyz = props['nxyz'][i]

        reps = []

        # if there are RDKit mols in the dataset, you can
        # get the 3D representation from the mol itself

        if 'rd_mols' in props:
            rd_mols = props['rd_mols'][i]
            for rd_mol in rd_mols:
                rep = torch.Tensor(get_3d_representation(xyz=None,
                                                         smiles=None,
                                                         method=method,
                                                         mol=rd_mol))
                reps.append(rep)

        # otherwise you can get the mols from the nxyz, but this
        # assumes the same ordering of nxyz and RDKit mol generated
        # from smiles, which triggers a warning

        else:
            nxyz_list = [nxyz]

            # if `mol_size` is there then split the nxyz into conformer
            # geomtries
            if 'mol_size' in props:
                mol_size = props['mol_size'][i].item()
                n_confs = nxyz.shape[0] // mol_size
                nxyz_list = torch.split(nxyz, [mol_size] * n_confs)

            for sub_nxyz in nxyz_list:

                msg = ("Warning: no RDKit mols found in dataset. "
                       "Using nxyz and SMILES and assuming that the "
                       "nxyz atom ordering is the same as in the RDKit "
                       "mol. Make sure to check this!")
                print(msg)

                xyz = sub_nxyz.detach().cpu().numpy().tolist()
                rep = torch.Tensor(get_3d_representation(xyz=xyz,
                                                         smiles=smiles,
                                                         method=method))
                reps.append(rep)

        reps = torch.stack(reps)
        dataset.props[method].append(reps)


def get_e3fp(mol, bits, smiles=None):
    """
    Get an E3FP fingerprint from an RDKit mol.
    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit mol object
        bits (int): Number of bits in fingerprint
    Returns:
        fp (np.array): fingerprint as numpy array
    """
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    mol.SetProp("_Name", smiles)
    fprint_params = {"bits": bits}
    fp = (fprints_from_mol(mol, fprint_params=fprint_params)[0]
          .to_vector().toarray().astype(int)
          ).reshape(-1)
    return fp


def add_e3fp(rd_dataset,
             fp_length,
             verbose=False,
             track=True):
    """
    Add E3FP fingerprints to each conformer in the dataset.
    Args:
        rd_dataset (nff.data.dataset): NFF dataset with RDKit mols.
        fp_length (int): length of each fingerprint
        verbose (bool): whether to print the progress made
        track (bool): whether to track progress with tqdm_enum
    Returns:
        rd_dataset (nff.data.dataset): NFF dataset updated with E3FP
    """

    # disable verbose logging from e3fp

    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True

    e3fp_list = []
    func = tqdm if (track) else lambda x: x

    for batch in func(rd_dataset):
        mols = batch["rd_mols"]
        smiles = batch["smiles"]
        fps = []
        for mol in mols:

            fp_array = get_e3fp(mol, fp_length, smiles)
            fps.append(torch.Tensor(fp_array))

        e3fp_list.append(torch.stack(fps))

    rd_dataset.props['e3fp'] = e3fp_list

    return rd_dataset
