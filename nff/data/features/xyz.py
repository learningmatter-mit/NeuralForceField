from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import torch


def get_conformer(xyz):
    n_atoms = len(xyz)
    conformer = Chem.Conformer(n_atoms)
    for i, nxyz in zip(range(n_atoms), xyz):
        conformer.SetAtomPosition(i, nxyz[1:])
    return conformer


def get_mol(xyz, smiles, with_conformer=True):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    if with_conformer:
        conformer = get_conformer(xyz)
    mol.AddConformer(conformer)

    return mol


def get_3d_representation(xyz, smiles, method):

    representation_fn = {
        'autocorrelation_3d': rdMD.CalcAUTOCORR3D,
        'rdf': rdMD.CalcRDF,
        'morse': rdMD.CalcMORSE,
        'whim': rdMD.CalcWHIM,
        'getaway': lambda x: rdMD.CalcWHIM(x, precision=0.001)
    }

    mol = get_mol(xyz=xyz, smiles=smiles)
    fn = representation_fn[method]

    return fn(mol)


def featurize_rdkit(dataset, method):
    dataset.props[method] = []
    props = dataset.props

    for nxyz, smiles in zip(props['nxyz'], props['smiles']):
        xyz = nxyz.detach().cpu().numpy().tolist()
        rep = get_3d_representation(xyz=xyz, smiles=smiles, method=method)
        dataset.props[method].append(torch.tensor(rep))
    dataset.props[method] = torch.stack(dataset.props[method])
