"""
Module for generating rdkit molobj/smiles/molecular graph from free atoms

Implementation by Jan H. Jensen, based on the paper

    Yeonjoon Kim and Woo Youn Kim
    "Universal Structure Conversion Method for Organic Molecules: From Atomic Connectivity
    to Three-Dimensional Geometry"
    Bull. Korean Chem. Soc. 2015, Vol. 36, 1769-1777
    DOI: 10.1002/bkcs.10334

"""

import copy
import errno
import itertools
import os
import pickle
import signal
from functools import wraps

try:
    from rdkit.Chem import rdEHTTools  # requires RDKit 2019.9.1 or later
except ImportError:
    rdEHTTools = None

from collections import defaultdict

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, GetPeriodicTable
from rdkit.Chem.rdchem import EditableMol

__ATOM_LIST__ = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v ",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
]


atomic_valence = defaultdict(list)
atomic_valence_electrons = {}
PERIODICTABLE = GetPeriodicTable()


for i in range(100):
    dics = [atomic_valence, atomic_valence_electrons]
    if all(i in dic for dic in dics):
        continue

    valence_list = list(PERIODICTABLE.GetValenceList(i))
    valence_num = PERIODICTABLE.GetNOuterElecs(i)

    atomic_valence[i] = valence_list
    atomic_valence_electrons[i] = valence_num


DEFAULT_SAVE = "mol.pickle"
# give up after 10 minutes
MAX_TIME = 600


class TimeoutError(Exception):
    pass


ERROR_MESSAGE = os.strerror(errno.ETIME)


def timeout(seconds, error_message=ERROR_MESSAGE):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def str_atom(atom):
    """
    convert integer atom to string atom
    """
    return __ATOM_LIST__[atom - 1]


def int_atom(atom):
    """
    convert str atom to integer atom
    """
    return __ATOM_LIST__.index(atom.lower()) + 1


def get_UA(maxValence_list, valence_list):
    """ """
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if not maxValence - valence > 0:
            continue
        UA.append(i)
        DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, use_graph=True):
    """ """
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = get_UA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, use_graph=use_graph)[0]

    return BO


def valences_not_too_large(BO, valences):
    """ """
    number_of_bonds_list = BO.sum(axis=1)
    return all(number_of_bonds <= valence for valence, number_of_bonds in zip(valences, number_of_bonds_list))


def BO_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atoms, valances, allow_charged_fragments=True):
    """
    Sanity of bond-orders

    args:
        BO -
        AC -
        charge -
        DU -


    optional
        allow_charges_fragments -


    returns:
        boolean - true of molecule is OK, false if not
    """

    if not valences_not_too_large(BO, valances):
        return False

    # total charge
    Q = 0

    # charge fragment list
    q_list = []

    if allow_charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atoms):
            q = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and charge > Q + 1:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    check_sum = (BO - AC).sum() == sum(DU)
    check_charge = charge == Q
    # check_len = len(q_list) <= abs(charge)

    return bool(check_sum and check_charge)


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    """ """

    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif (atom == 15 and BO_valence == 5) or (atom == 16 and BO_valence == 6):
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    """
    This hack should not be needed anymore, but is kept just in case

    """

    Chem.SanitizeMol(mol)

    # rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
    #              '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
    #              '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
    #              '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
    #              '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
    #              '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    rxn_smarts = [
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][CX3-,NX3-:5][#6,#7:6]1=[#6,#7:7]>>"
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]=[#6,#7:4][-0,-0:5]=[#6,#7:6]1[#6-,#7-:7]",
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3](=[#6,#7:4])[#6,#7:5]=[#6,#7:6][CX3-,NX3-:7]1>>"
        "[#6,#7:1]1=[#6,#7:2][#6,#7:3]([#6-,#7-:4])=[#6,#7:5][#6,#7:6]=[-0,-0:7]1",
    ]

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
                Chem.SanitizeMol(fragment)
        mol = fragment if i == 0 else Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(mol, BO_matrix, atoms, atomic_valence_electrons, mol_charge, allow_charged_fragments=True):
    """
    based on code written by Paolo Toscani

    From bond order, atoms, valence structure and total charge, generate an
    rdkit molecule.

    args:
        mol - rdkit molecule
        BO_matrix - bond order matrix of molecule
        atoms - list of integer atomic symbols
        atomic_valence_electrons -
        mol_charge - total charge of molecule

    optional:
        allow_charged_fragments - bool - allow charged fragments

    returns
        mol - updated rdkit molecule with bond connectivity

    """

    l1 = len(BO_matrix)
    l2 = len(atoms)
    BO_valences = list(BO_matrix.sum(axis=1))

    if l1 != l2:
        raise RuntimeError(f"sizes of adjMat ({l1:d}) and Atoms {l2:d} differ")

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

    for i in range(l1):
        for j in range(i + 1, l1):
            bo = int(round(BO_matrix[i, j]))
            if bo == 0:
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)

    mol = rwMol.GetMol()

    if allow_charged_fragments:
        mol = set_atomic_charges(mol, atoms, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge)
    else:
        mol = set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences)

    return mol


def set_atomic_charges(mol, atoms, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge):
    """ """
    q = 0
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if abs(charge) > 0:
            a.SetFormalCharge(int(charge))

    mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol, atoms, atomic_valence_electrons, BO_valences):
    """

    The number of radical electrons = absolute atomic charge

    """
    for i, atom in enumerate(atoms):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(atom, atomic_valence_electrons[atom], BO_valences[i])

        if abs(charge) > 0:
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    """ """
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k + 1 :]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))  # noqa

    return bonds


def get_UA_pairs(UA, AC, use_graph=True):
    """ """

    bonds = get_bonds(UA, AC)

    if len(bonds) == 0:
        return [()]

    if use_graph:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA) / 2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]

        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """

    implementation of the algorithm shown in Figure 2

    UA: unsaturated atoms

    DU: degree of unsaturation (u matrix in Figure)

    best_BO: Bcurr in Figure

    """

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    AC_valence = list(AC.sum(axis=1))
    valences_list_of_lists = [atomic_valence[atomicNum] for atomicNum in atoms]

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = itertools.product(*valences_list_of_lists)

    best_BO = AC.copy()

    for valences in valences_list:
        UA, DU_from_AC = get_UA(valences, AC_valence)

        check_len = len(UA) == 0
        if check_len:
            check_bo = BO_is_OK(
                AC,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
            )
        else:
            check_bo = None

        if check_len and check_bo:
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, use_graph=use_graph)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, use_graph=use_graph)
            status = BO_is_OK(
                BO,
                AC,
                charge,
                DU_from_AC,
                atomic_valence_electrons,
                atoms,
                valences,
                allow_charged_fragments=allow_charged_fragments,
            )

            if status:
                return BO, atomic_valence_electrons

            if BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2mol(mol, AC, atoms, charge, allow_charged_fragments=True, use_graph=True):
    """ """

    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons = AC2BO(
        AC, atoms, charge, allow_charged_fragments=allow_charged_fragments, use_graph=use_graph
    )

    # add BO connectivity and charge info to mol object
    mol = BO2mol(mol, BO, atoms, atomic_valence_electrons, charge, allow_charged_fragments=allow_charged_fragments)

    return mol


def get_proto_mol(atoms):
    """ """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def read_xyz_file(filename, look_for_charge=True):
    """ """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atoms = [int_atom(atom) for atom in atomic_symbols]

    return atoms, charge, xyz_coordinates


def xyz2AC(atoms, xyz, charge, use_huckel=False):
    """

    atoms and coordinates to atom connectivity (AC)

    args:
        atoms - int atom types
        xyz - coordinates
        charge - molecule charge

    optional:
        use_huckel - Use Huckel method for atom connecitivty

    returns
        ac - atom connectivity matrix
        mol - rdkit molecule

    """

    if use_huckel:
        return xyz2AC_huckel(atoms, xyz, charge)
    return xyz2AC_vdW(atoms, xyz)


def xyz2AC_vdW(atoms, xyz):
    # Get mol template
    mol = get_proto_mol(atoms)

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    AC = get_AC(mol)

    return AC, mol


def get_AC(mol, covalent_factor=1.3):
    """

    Generate adjacent matrix from atoms and coordinates.

    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not


    covalent_factor - 1.3 is an arbitrary factor

    args:
        mol - rdkit molobj with 3D conformer

    optional
        covalent_factor - increase covalent bond length threshold with facto

    returns:
        AC - adjacent matrix

    """

    # Calculate distance matrix
    dMat = Chem.Get3DDistanceMatrix(mol)

    pt = Chem.GetPeriodicTable()
    num_atoms = mol.GetNumAtoms()
    AC = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum()) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum()) * covalent_factor
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC


def xyz2AC_huckel(atomicNumList, xyz, charge):
    """

    args
        atomicNumList - atom type list
        xyz - coordinates
        charge - molecule charge

    returns
        ac - atom connectivity
        mol - rdkit molecule

    """
    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms, num_atoms)).astype(int)

    mol_huckel = Chem.Mol(mol)
    # mol charge arbitrarily added to 1st atom
    mol_huckel.GetAtomWithIdx(0).SetFormalCharge(charge)

    passed, result = rdEHTTools.RunMol(mol_huckel)
    opop = result.GetReducedOverlapPopulationMatrix()
    tri = np.zeros((num_atoms, num_atoms))
    tri[np.tril(np.ones((num_atoms, num_atoms), dtype=bool))] = opop  # lower triangular to square matrix
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair_pop = abs(tri[j, i])
            if pair_pop >= 0.15:  # arbitry cutoff for bond. May need adjustment
                AC[i, j] = 1
                AC[j, i] = 1

    return AC, mol


def chiral_stereo_check(mol):
    """
    Find and embed chiral information into the model based on the coordinates

    args:
        mol - rdkit molecule, with embeded conformer

    """
    Chem.SanitizeMol(mol)
    Chem.DetectBondStereochemistry(mol, -1)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)

    return


def check_mol(mol, coordinates):
    conf = mol.GetConformers()[0]
    new_coords = conf.GetPositions()
    old_coords = np.array(coordinates)

    delta = np.linalg.norm(new_coords - old_coords)
    if delta < 1e-2:
        return mol

    # sometimes the positions of the atoms get rearranged for some reason

    new_pos = mol.GetConformers()[0].GetPositions()

    dist = np.linalg.norm(
        new_pos.reshape(1, *new_pos.shape) - old_coords.reshape(old_coords.shape[0], 1, old_coords.shape[1]), axis=-1
    )

    new_idx = dist.argmin(-1).tolist()
    rev_idx = dist.argmin(0).tolist()

    ed_mol = EditableMol(Chem.MolFromSmiles(""))

    for idx in new_idx:
        atom = mol.GetAtoms()[idx]
        ed_mol.AddAtom(atom)

    all_old_bond_idx = []
    all_old_bond_types = []

    for atom in mol.GetAtoms():
        bonds = atom.GetBonds()
        old_bond_idx = [[i.GetBeginAtomIdx(), i.GetEndAtomIdx()] for i in bonds]
        bond_types = [i.GetBondType() for i in bonds]

        use_idx = [j for j, idx in enumerate(old_bond_idx) if idx not in all_old_bond_idx]

        all_old_bond_idx += [old_bond_idx[j] for j in use_idx]
        all_old_bond_types += [bond_types[j] for j in use_idx]

    for bond_idx, bond_type in zip(all_old_bond_idx, all_old_bond_types):
        new_bond_idx = [rev_idx[bond_idx[0]], rev_idx[bond_idx[1]]]

        ed_mol.AddBond(new_bond_idx[0], new_bond_idx[1], bond_type)

    new_mol = ed_mol.GetMol()

    # add the conformer

    conformer = Chem.Conformer(len(coordinates))
    for i, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(i, xyz)
    new_mol.AddConformer(conformer)

    return new_mol


@timeout(seconds=MAX_TIME)
def xyz2mol(
    atoms, coordinates, charge=0, allow_charged_fragments=True, use_graph=True, use_huckel=False, embed_chiral=True
):
    """
    Generate a rdkit molobj from atoms, coordinates and a total_charge.

    args:
        atoms - list of atom types (int)
        coordinates - 3xN Cartesian coordinates
        charge - total charge of the system (default: 0)

    optional:
        allow_charged_fragments - alternatively radicals are made
        use_graph - use graph (networkx)
        use_huckel - Use Huckel method for atom connectivity prediction
        embed_chiral - embed chiral information to the molecule

    returns:
        mol - rdkit molobj

    """

    # Get atom connectivity (AC) matrix, list of atomic numbers, molecular charge,
    # and mol object with no connectivity information
    AC, mol = xyz2AC(atoms, coordinates, charge, use_huckel=use_huckel)

    # Convert AC to bond order matrix and add connectivity and charge info to
    # mol object
    new_mol = AC2mol(mol, AC, atoms, charge, allow_charged_fragments=allow_charged_fragments, use_graph=use_graph)

    # Check for stereocenters and chiral centers
    if embed_chiral:
        chiral_stereo_check(new_mol)

    new_mol = check_mol(mol=new_mol, coordinates=coordinates)

    return new_mol


def main():
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(usage="%(prog)s [options] molecule.xyz")
    parser.add_argument("structure", metavar="structure", type=str)
    parser.add_argument("-s", "--sdf", action="store_true", help="Dump sdf file")
    parser.add_argument("--ignore-chiral", action="store_true", help="Ignore chiral centers")
    parser.add_argument("--no-charged-fragments", action="store_true", help="Allow radicals to be made")
    parser.add_argument("--no-graph", action="store_true", help="Run xyz2mol without networkx dependencies")

    # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
    # otherwise van der Waals radii are used
    parser.add_argument("--use-huckel", action="store_true", help="Use Huckel method for atom connectivity")
    parser.add_argument(
        "-o", "--output-format", action="store", type=str, help="Output format [smiles,sdf] (default=sdf)"
    )
    parser.add_argument("-c", "--charge", action="store", metavar="int", type=int, help="Total charge of the system")
    parser.add_argument("--save_name", type=str, default=DEFAULT_SAVE, help="Save name for RDKit mol")

    args = parser.parse_args()

    # read xyz file
    filename = args.structure

    # allow for charged fragments, alternatively radicals are made
    charged_fragments = not args.no_charged_fragments

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = not args.no_graph

    # chiral comment
    embed_chiral = not args.ignore_chiral

    # read atoms and coordinates. Try to find the charge
    atoms, charge, xyz_coordinates = read_xyz_file(filename)

    # huckel uses extended Huckel bond orders to locate bonds (requires RDKit 2019.9.1 or later)
    # otherwise van der Waals radii are used
    use_huckel = args.use_huckel

    # if explicit charge from args, set it
    if args.charge is not None:
        charge = int(args.charge)

    # Get the molobj
    mol = xyz2mol(
        atoms,
        xyz_coordinates,
        charge=charge,
        use_graph=quick,
        allow_charged_fragments=charged_fragments,
        embed_chiral=embed_chiral,
        use_huckel=use_huckel,
    )

    # Print output
    if args.output_format == "sdf":
        txt = Chem.MolToMolBlock(mol)
        print(txt)

    else:
        # Canonical hack
        isomeric_smiles = not args.ignore_chiral
        smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
        m = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(m, isomericSmiles=isomeric_smiles)
        print(smiles)

    save_name = args.save_name
    with open(save_name, "wb") as f:
        pickle.dump(mol, f)
