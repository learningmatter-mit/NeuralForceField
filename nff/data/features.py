import torch
import numpy as np
import copy
import time
from tqdm import tqdm
from rdkit import Chem
from nff.utils.xyz2mol import xyz2mol

QUICK = True
EMBED_CHIRAL = True
USE_HUCKEL = False
CHARGED_FRAGMENTS = True


BOND_FEAT_TYPES = ["bond_type",
                   "conjugated",
                   "in_ring",
                   "stereo",
                   "in_ring_size"]

ATOM_FEAT_TYPES = ["atom_type",
                   "num_bonds",
                   "formal_charge",
                   "chirality",
                   "num_bonded_h",
                   "hybrid",
                   "aromaticity",
                   "mass"]

CHIRAL_OPTIONS = ["chi_unspecified",
                  "chi_tetrahedral_cw",
                  "chi_tetrahedral_ccw",
                  "chi_other"]

HYBRID_OPTIONS = ["s",
                  "sp",
                  "sp2",
                  "sp3",
                  "sp3d",
                  "sp3d2"]

BOND_OPTIONS = ["single",
                "double",
                "triple",
                "aromatic"]

STEREO_OPTIONS = ["stereonone",
                  "stereoany",
                  "stereoz",
                  "stereoe",
                  "stereocis",
                  "stereotrans"]

MAX_AT_NUM = 100
AT_NUM = list(range(1, MAX_AT_NUM + 1))
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
BONDS = [0, 1, 2, 3, 4, 5]
NUM_H = [0, 1, 2, 3, 4]
RING_SIZE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def remove_bad_idx(dataset, smiles_list, bad_idx, verbose=True):
    bad_idx = sorted(list(set(bad_idx)))
    new_props = {}
    for key, values in dataset.props.items():
        new_props[key] = [val for i, val in enumerate(
            values) if i not in bad_idx]
        if not new_props[key]:
            continue
        if type(values) is torch.Tensor:
            new_props[key] = torch.stack(new_props[key])

    dataset.props = new_props

    total_len = len(smiles_list)
    good_len = total_len - len(bad_idx)
    conv_pct = good_len / total_len * 100

    if verbose:
        print(("Converted %d of %d "
           "species (%.2f%%)" % (
               good_len, total_len, conv_pct)))


def smiles_from_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    new_smiles = Chem.MolToSmiles(mol)
    new_mol = Chem.MolFromSmiles(new_smiles)
    new_smiles = Chem.MolToSmiles(new_mol)

    return new_smiles, new_mol


def smiles_from_mol(mol):

    new_smiles = Chem.MolToSmiles(mol)
    new_mol = Chem.MolFromSmiles(new_smiles)
    new_smiles = Chem.MolToSmiles(new_mol)

    return new_smiles, new_mol


def get_undirected_bonds(mol):

    bond_list = []
    bonds = mol.GetBonds()

    for bond in bonds:

        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        lower = min((start, end))
        upper = max((start, end))

        bond_list.append([lower, upper])

    return bond_list


def check_connectivity(mol_0, mol_1):

    bonds_0 = get_undirected_bonds(mol_0)
    bonds_1 = get_undirected_bonds(mol_1)

    return bonds_0 == bonds_1


def verify_smiles(rd_mol, smiles):

    rd_smiles, new_rd_mol = smiles_from_mol(rd_mol)
    db_smiles, db_mol = smiles_from_smiles(smiles)

    if rd_smiles == db_smiles:
        return

    # try ignoring stereochemistry

    Chem.RemoveStereochemistry(new_rd_mol)
    rd_smiles, new_rd_mol = smiles_from_mol(new_rd_mol)

    Chem.RemoveStereochemistry(db_mol)
    db_smiles, db_mol = smiles_from_mol(db_mol)

    if rd_smiles == db_smiles:
        return

    # try checking bond connectivity

    good_con = check_connectivity(mol_0=new_rd_mol,
                                  mol_1=db_mol)

    if good_con:
        msg = (("WARNING: xyz2mol SMILES is {} "
                "and database SMILES is {}. "
                "However, the connectivity is the same. "
                "Check to make sure the SMILES are resonances "
                "structures.".format(rd_smiles, db_smiles)))
        return

    # otherwise raise an exception

    msg = (("SMILES created by xyz2mol is {}, "
            "which doesn't match the database "
            "SMILES {}.".format(rd_smiles, db_smiles)))
    raise Exception(msg)


def log_failure(bad_idx, i):

    if i == 0:
        return

    good_len = i - len(bad_idx)
    conv_pct = good_len / i * 100

    print(("Converted %d of %d "
           "species (%.2f%%)" % (
               good_len, i, conv_pct)))


def log_missing(missing_e):
    if not missing_e:
        print("No elements are missing from xyz2mol")
    else:
        missing_e = list(set(missing_e))
        print("Elements {} are missing from xyz2mol".format(
            ", ".join(missing_e)))


def make_rd_mols(dataset, verbose=True):

    num_atoms = dataset.props['num_atoms']
    mol_size = dataset.props.get("mol_size", num_atoms).tolist()
    smiles_list = dataset.props["smiles"]
    all_nxyz = dataset.props["nxyz"]
    charges = dataset.props["charge"]
    dataset.props["rd_mols"] = []

    all_mols = []
    bad_idx = []

    # for i, smiles in tqdm(enumerate(smiles_list)):

    for i, smiles in enumerate(smiles_list):

        num_confs = (num_atoms[i] // mol_size[i]).item()
        split_sizes = [mol_size[i]] * num_confs
        nxyz_list = torch.split(all_nxyz[i], split_sizes)
        charge = charges[i]

        spec_mols = []
        missing_e = []

        for j, nxyz in enumerate(nxyz_list):

            if i in bad_idx:
                continue

            xyz = nxyz[:, 1:].tolist()
            atoms = nxyz[:, 0].numpy().astype('int').tolist()

            try:

                mol = xyz2mol(atoms=atoms,
                              coordinates=xyz,
                              charge=charge,
                              use_graph=QUICK,
                              allow_charged_fragments=CHARGED_FRAGMENTS,
                              embed_chiral=EMBED_CHIRAL,
                              use_huckel=USE_HUCKEL)
                verify_smiles(rd_mol=mol, smiles=smiles)

            except Exception as e:

                print(("xyz2mol failed "
                       "with error '{}' ".format(e)))
                print("Removing smiles {}".format(smiles))
                bad_idx.append(i)

                if verbose:
                    log_failure(bad_idx=bad_idx, i=i)

                if str(e).isdigit():
                    missing_e.append(int(str(e)))

                continue

            spec_mols.append(mol)

        all_mols.append(spec_mols)
        dataset.props["rd_mols"].append(spec_mols)

    remove_bad_idx(dataset=dataset,
                   smiles_list=smiles_list,
                   bad_idx=bad_idx,
                   verbose=verbose)

    if verbose:
        log_missing(missing_e)


    return dataset


def make_one_hot(options, result):

    index = options.index(result) if result in options else -1
    one_hot = torch.zeros(len(options) + 1)
    one_hot[index] = 1

    return one_hot


def get_bond_features(bond, feat_type):

    if feat_type == "conjugated":
        conj = bond.GetIsConjugated()
        result = torch.Tensor([conj])

        return result

    elif feat_type == "bond_type":
        bond_type = bond.GetBondType().name.lower()
        options = BOND_OPTIONS
        one_hot = make_one_hot(options=options,
                               result=bond_type)
        return one_hot

    elif feat_type == "in_ring_size":
        options = RING_SIZE
        ring_size = - 1
        for option in options:
            is_in_size = bond.IsInRingSize(option)
            if is_in_size:
                ring_size = option
                break

        one_hot = make_one_hot(options=options,
                               result=ring_size)

        return one_hot

    if feat_type == "in_ring":
        in_ring = bond.IsInRing()
        result = torch.Tensor([in_ring])
        return result

    elif feat_type == "stereo":
        stereo = bond.GetStereo().name.lower()
        options = STEREO_OPTIONS
        one_hot = make_one_hot(options=options,
                               result=stereo)
        return one_hot


def get_atom_features(atom, feat_type):

    if feat_type == "atom_type":
        atom_num = atom.GetAtomicNum() - 1
        options = list(range(MAX_AT_NUM))
        one_hot = make_one_hot(options=options,
                               result=atom_num)
        return one_hot

    elif feat_type == "num_bonds":
        num_bonds = atom.GetTotalDegree()
        options = BONDS
        one_hot = make_one_hot(options=options,
                               result=num_bonds)
        return one_hot

    elif feat_type == "formal_charge":

        fc = atom.GetFormalCharge()
        options = FORMAL_CHARGES
        one_hot = make_one_hot(options=options,
                               result=fc)

        return one_hot

    elif feat_type == "chirality":
        chirality = atom.GetChiralTag().name.lower()
        options = CHIRAL_OPTIONS
        one_hot = make_one_hot(options=options,
                               result=chirality)
        return one_hot

    elif feat_type == "num_bonded_h":

        neighbors = [at.GetAtomicNum() for at
                     in atom.GetNeighbors()]
        num_h = len([i for i in neighbors if
                     i == 1])

        options = NUM_H
        one_hot = make_one_hot(options=options,
                               result=num_h)
        return one_hot

    elif feat_type == "hybrid":
        hybrid = atom.GetHybridization().name.lower()
        options = HYBRID_OPTIONS
        one_hot = make_one_hot(options=options,
                               result=hybrid)
        return one_hot

    elif feat_type == "aromaticity":
        aromatic = atom.GetIsAromatic()
        one_hot = torch.Tensor([aromatic])
        return one_hot

    elif feat_type == "mass":
        mass = atom.GetMass() / 100
        result = torch.Tensor([mass])
        return result


def get_all_bond_feats(bond, feat_types):

    feat_dic = {}
    for feat_type in feat_types:
        feature = get_bond_features(bond=bond,
                                    feat_type=feat_type)
        feat_dic[feat_type] = feature
    return feat_dic


def get_all_atom_feats(atom, feat_types):

    feat_dic = {}
    for feat_type in feat_types:
        feature = get_atom_features(atom=atom,
                                    feat_type=feat_type)
        feat_dic[feat_type] = feature
    return feat_dic


def featurize_bonds(dataset, feat_types=BOND_FEAT_TYPES):

    props = dataset.props

    props["bond_list"] = []
    props["bond_features"] = []
    props["num_bonds"] = []

    num_atoms = dataset.props['num_atoms']
    mol_size = dataset.props.get("mol_size", num_atoms).tolist()

    for i, rd_mols in enumerate(dataset.props["rd_mols"]):

        num_confs = (num_atoms[i] // mol_size[i]).item()
        split_sizes = [mol_size[i]] * num_confs

        props["bond_list"].append([])
        props["num_bonds"].append([])

        all_props = []

        for j, rd_mol in enumerate(rd_mols):

            bonds = rd_mol.GetBonds()
            bond_list = []

            for bond in bonds:

                all_props.append(torch.tensor([]))

                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                lower = min((start, end))
                upper = max((start, end))

                bond_list.append([lower, upper])
                feat_dic = get_all_bond_feats(bond=bond,
                                              feat_types=feat_types)

                for key, feat in feat_dic.items():
                    all_props[-1] = torch.cat((all_props[-1], feat))

            other_atoms = sum(split_sizes[:j])
            shifted_bond_list = np.array(bond_list) + other_atoms

            props["bond_list"][-1].append(torch.LongTensor(
                shifted_bond_list))
            props["num_bonds"][-1].append(len(bonds))

        props["bond_list"][-1] = torch.cat(props["bond_list"][-1])
        props["bond_features"].append(torch.stack(all_props))

    return dataset

def featurize_atoms(dataset, feat_types=ATOM_FEAT_TYPES):

    props = dataset.props
    props["atom_features"] = []

    for i, rd_mols in enumerate(dataset.props["rd_mols"]):

        all_props = []
        for rd_mol in rd_mols:
            atoms = rd_mol.GetAtoms()

            for atom in atoms:
                all_props.append(torch.tensor([]))
                feat_dic = get_all_atom_feats(atom=atom,
                                              feat_types=feat_types)
                for key, feat in feat_dic.items():
                    all_props[-1] = torch.cat((all_props[-1], feat))

        props["atom_features"].append(torch.stack(all_props))

    return dataset

def decode_one_hot(options, vector):

    if options == [bool]:
        return bool(vector.item())
    elif options == [float]:
        return vector.item()

    index = vector.nonzero()
    if len(index) == 0:
        result = None
    else:
        result = options[index]

    return result


def decode_atomic(features):

    options_list = [AT_NUM,
                    BONDS,
                    FORMAL_CHARGES,
                    CHIRAL_OPTIONS,
                    NUM_H,
                    HYBRID_OPTIONS,
                    [bool],
                    [float]]

    indices = []
    for i, item in enumerate(options_list):
        indices.append(len(item))
        if item not in [[bool], [float]]:
            indices[-1] += 1

    vectors = torch.split(features, indices)

    dic = {}
    for i, vector in enumerate(vectors):
        options = options_list[i]
        name = ATOM_FEAT_TYPES[i]
        result = decode_one_hot(options=options,
                                vector=vector)
        dic[name] = result
        if name == "mass":
            dic[name] *= 100

    return dic


def decode_bond(features):

    options_list = [BOND_OPTIONS,
                    [bool],
                    [bool],
                    STEREO_OPTIONS,
                    RING_SIZE]

    indices = []
    for i, item in enumerate(options_list):
        indices.append(len(item))
        if item not in [[bool], [float]]:
            indices[-1] += 1

    vectors = torch.split(features, indices)
    dic = {}

    for i, vector in enumerate(vectors):
        options = options_list[i]
        name = BOND_FEAT_TYPES[i]

        result = decode_one_hot(options=options,
                                vector=vector)
        dic[name] = result

    return dic


def featurize_dataset(dataset,
                      bond_feats=BOND_FEAT_TYPES,
                      atom_feats=ATOM_FEAT_TYPES):

    print("Converting xyz to RDKit mols...")
    dataset = make_rd_mols(dataset)
    print("Completed conversion to RDKit mols.")

    print("Featurizing bonds...")
    dataset = featurize_bonds(dataset, feat_types=bond_feats)
    print("Completed featurizing bonds.")

    print("Featurizing atoms...")
    dataset = featurize_atoms(dataset, feat_types=atom_feats)
    print("Completed featurizing atoms.")

    props = dataset.props
    props.pop("rd_mols")
    props["bonded_nbr_list"] = copy.deepcopy(props["bond_list"])
    props.pop("bond_list")
