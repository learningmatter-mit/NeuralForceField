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


def get_3d_representation(xyz, smiles, method, mol=None):

    representation_fn = {
        'autocorrelation_3d': rdMD.CalcAUTOCORR3D,
        'rdf': rdMD.CalcRDF,
        'morse': rdMD.CalcMORSE,
        'whim': rdMD.CalcWHIM,
        'getaway': lambda x: rdMD.CalcWHIM(x, precision=0.001)
    }

    if mol is None:
        mol = get_mol(xyz=xyz, smiles=smiles)
    fn = representation_fn[method]

    return fn(mol)


def featurize_rdkit(dataset, method):
    dataset.props[method] = []
    props = dataset.props

    for i in range(len(props['nxyz'])):

        smiles = props['smiles'][i]
        nxyz = props['nxyz'][i]

        reps = []

        if 'rd_mols' in props:
            rd_mols = props['rd_mols'][i]
            for rd_mol in rd_mols:
                rep = torch.Tensor(get_3d_representation(xyz=None,
                                                         smiles=None,
                                                         method=method,
                                                         mol=rd_mol))
                reps.append(rep)
        else:
            nxyz_list = [nxyz]
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
