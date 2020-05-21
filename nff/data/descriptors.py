from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD
import torch
import math
from scipy.special import factorial as fact
from scipy.special import lpmv as legendre
from scipy.special import eval_genlaguerre as laguerre

from torch.utils.data import DataLoader
from nff.data.loader import collate_dicts
from nff.utils.cuda import batch_to
from nff.io.ase import AtomsBatch


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


def spher_harmonic(theta, phi, l, m):
    """ 
    Torch implementation of spherical harmonics, returning the real
    and imaginary parts separately. 
    Args:
        theta (torch.tensor): tensor of polar angles
        phi (torch.tensor): tensor of azimuthal angles
        l (int): angular momentum quantum number
        m (int): magnetic quantum number 
    Returns:
        y_r (torch.tensor): tensor of the real parts of the spherical
            harmonics
        y_i (torch.tensor): tensor of the imaginary parts of the spherical
            harmonics
    """

    # prefactor
    pref = ((2 * l + 1)/(4 * math.pi) * fact(l - m) / fact(l + m)) ** 0.5
    cos_t = torch.cos(theta)

    # Calculate the real and imaginary parts
    # Note thathe Condon-Shortley phase already included in the definition
    # of the Legendre polynomials

    device = cos_t.device
    y_r = pref * legendre(m, l, cos_t.cpu()) * torch.cos(m * phi.cpu())
    y_i = pref * legendre(m, l, cos_t.cpu()) * torch.sin(m * phi.cpu())

    return y_r.to(device), y_i.to(device)


def harmonic_coef(theta, phi, l, m):
    """
    Coefficient c_lm weighting a spherical harmonic for coordinates (theta, phi).
    Args:
        theta (torch.tensor): tensor of polar angles
        phi (torch.tensor): tensor of azimuthal angles
        l (int): angular momentum quantum number
        m (int): magnetic quantum number  
    Returns:
        c_r (torch.tensor): tensor of the real parts of the weighting coefficients
        c_i (torch.tensor): tensor of the imaginary parts of the weighting coefficients
            harmonics
    """

    y_r, y_i = spher_harmonic(theta=theta, phi=phi, l=l, m=m)

    # the weighting coefficient is Y_{lm}^*, so the real part
    # is just y_r, while the imaginary part is -y_i

    c_r = y_r
    c_i = -y_i

    return c_r, c_i


def radial_coef(r, a0, n, l):
    """
    Get the radial component of the expansion coefficient c_nlm.
    Args:
        r (torch.tensor): tensor of radial distances
        a0 (float): "Bohr radius": characteristic length scale
        n (int): principal quantum number
        l (int): angular momentum quantum number
    returns:
        coef (float): radial component fo the coefficient
    """

    # prefactor
    pref = (((2 / (n * a0)) ** 3) * fact(n - l - 1) /
            (2 * n * fact(n + l))) ** 0.5
    # dimensionless distance rho
    rho = 2 * r / (n * a0)

    device = rho.device
    rho_cpu = rho.cpu()
    coef = pref * torch.exp(-rho_cpu / 2) * rho_cpu ** l * laguerre(
        n - l - 1, 2 * l + 1, rho_cpu)

    return coef.to(device)


def get_spher_coords(xyz):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Args:
        xyz (torch.tensor): x, y, and z coordinates of atoms in 
            a molecule
    Returns:
        r (torch.tensor): tensor of radial distances
        theta (torch.tensor): tensor of polar angles
        phi (torch.tensor): tensor of azimuthal angles
    """

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = (xyz ** 2).sum(-1) ** 0.5

    # make sure to use atan2 and not just arctan(y/x)
    # because otherwise it can give you the wrong
    # quadrant for phi, which makes the answer wrong

    phi = torch.atan2(y, x)
    theta = torch.acos(z / r)

    return r, theta, phi


def norm_and_sum(batch_xyz, n, l, a0, num_atoms):
    r"""
    Multiply every expansion coefficient c_nlm by its complex
    conjugate and sum over m to make it rotationally invariant.
    Args:
        batch_xyz (torch.tensor): concatenated xyz of multiple
            molecules in a batch.
        n (int): principal quantum number
        l (int): angular momentum quantum number
        a0 (float): "Bohr radius": characteristic length scale
        num_atoms (list): number of atoms in each molecule
            in the batch.
    Returns:
        m_norm (torch.tensor): \sum_m |c_{nlm}|^2.

    """

    # initialize to 0

    m_norm = 0

    # sum over m from -l to 1 in integer steps

    for m in range(-l, l + 1):

        r, theta, phi = get_spher_coords(batch_xyz)

        # get the l, m components of the coefficient
        spher_r, spher_i = harmonic_coef(theta=theta,
                                         phi=phi,
                                         l=l,
                                         m=m,
                                         )

        # get the radial n, l component
        rad_comp = radial_coef(r=r,
                               a0=a0,
                               n=n,
                               l=l)

        # the coefficient for the entire molecule
        # sums over all atom positions in a molecule.
        # Need to first split by `num_atoms` to group
        # by molecule, then sum over each atom in that
        # molecule.

        cm_r_by_atom = torch.split(spher_r * rad_comp, num_atoms)
        cm_r = torch.stack([item.sum(0) for item in cm_r_by_atom])

        cm_i_by_atom = torch.split(spher_i * rad_comp, num_atoms)
        cm_i = torch.stack([item.sum(0) for item in cm_i_by_atom])

        # add |c_{nlm}|^2 to the final result

        m_norm += cm_r ** 2 + cm_i ** 2

    return m_norm


def make_hydrogenic_rep(batch_nxyz, n_max, a0, num_atoms, atom_type):
    r"""
    Represent the molecule through an expansion of Hydrogen-like
    wave functions.
    Args:
        xyz (torch.tensor): x, y, and z coordinates of atoms in 
            a molecule
        n_max (int): highest principal quantum number
        a0 (float): "Bohr radius": characteristic length scale
    Returns:
        feats (torch.tensor): n x (n+1)/ 2 dimensional feature vector.

    """

    feats = []
    if atom_type is None:
        batch_xyz = batch_nxyz[:, 1:]
        mask_num_atoms = num_atoms
    else:
        mask = batch_nxyz[:, 0] == atom_type
        batch_xyz = batch_nxyz[mask][:, 1:]
        mask_num_atoms = [item.sum().item() for item in torch.split(
            mask, num_atoms)]

    for n in range(1, n_max + 1):
        for l in range(n):
            out = norm_and_sum(batch_xyz=batch_xyz,
                               n=n,
                               l=l,
                               a0=a0,
                               num_atoms=mask_num_atoms)
            feats.append(out)
    feats = torch.stack(feats, dim=-1)

    return feats


def center_atoms(dataset):

    for i in range(len(dataset)):
        nxyz = dataset.props["nxyz"][i]
        numbers = nxyz[:, 0].cpu().numpy()
        positions = nxyz[:, 1:].cpu().numpy()
        atoms = AtomsBatch(numbers=numbers,
                           positions=positions)
        com = atoms.get_center_of_mass()

        dataset.props["nxyz"][i][:, 1:] -= com


def featurize_hydrogenic(dataset,
                         n_max,
                         a0,
                         atom_types,
                         device='cpu',
                         batch_size=1000):

    loader = DataLoader(dataset, batch_size=batch_size,
                        collate_fn=collate_dicts)
    all_feats = []
    num_atom_types = len(atom_types)
    print("Featurizing with %d atom types" % num_atom_types)

    center_atoms(dataset)

    for atom_type in atom_types:
        feats = []
        for i, batch in enumerate(loader):

            batch = batch_to(batch, device)
            batch_nxyz = batch["nxyz"]
            num_atoms = batch["num_atoms"].detach().cpu().tolist()
            new_feats = make_hydrogenic_rep(batch_nxyz=batch_nxyz,
                                            n_max=n_max,
                                            a0=a0,
                                            num_atoms=num_atoms,
                                            atom_type=atom_type)

            feats.append(new_feats)

            print("Batch %d complete" % i)

        feats = torch.cat(feats)
        all_feats.append(feats)
    dataset.props["hydro_feats"] = torch.cat(all_feats, dim=-1)
